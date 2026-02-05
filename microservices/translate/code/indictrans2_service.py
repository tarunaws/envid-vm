import os
import time
import threading
import gc
import psutil
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - runtime dependency
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

app = FastAPI(title="IndicTrans2 Service", version="1.0")

LANG_MAP: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "as": "asm_Beng",
    "ne": "npi_Deva",
    "ks": "kas_Arab",
    "kok": "gom_Deva",
    "doi": "doi_Deva",
    "sd": "snd_Arab",
    "sa": "san_Deva",
    "mai": "mai_Deva",
}

LANG_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "as": "Assamese",
    "ne": "Nepali",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "doi": "Dogri",
    "sd": "Sindhi",
    "sa": "Sanskrit",
    "mai": "Maithili",
}

MODEL_EN_INDIC = os.getenv("INDIC_TRANS_MODEL_EN_INDIC", "ai4bharat/indictrans2-en-indic-1B")
MODEL_INDIC_EN = os.getenv("INDIC_TRANS_MODEL_INDIC_EN", "ai4bharat/indictrans2-indic-en-1B")
MODEL_INDIC_INDIC = os.getenv("INDIC_TRANS_MODEL_INDIC_INDIC", "ai4bharat/indictrans2-indic-indic-1B")
MODEL_DEFAULT = os.getenv("INDIC_TRANS_MODEL_DEFAULT", "").strip()

BATCH_SIZE = int(os.getenv("INDIC_TRANS_BATCH_SIZE") or 8)  # Reduced from 16
MAX_TOKENS = int(os.getenv("INDIC_TRANS_MAX_TOKENS") or 512)
CACHE_DIR = (os.getenv("INDIC_TRANS_CACHE_DIR") or "").strip() or None
MAX_TEXT_LENGTH = int(os.getenv("INDIC_TRANS_MAX_TEXT_LENGTH") or 4000)  # Prevent OOM
MAX_CONCURRENT_REQUESTS = int(os.getenv("INDIC_TRANS_MAX_CONCURRENT") or 1)  # Limit parallel translations
MEMORY_LIMIT_PCT = float(os.getenv("INDIC_TRANS_MEMORY_LIMIT_PCT") or 85.0)  # Trigger cleanup at 85%
REQUEST_TIMEOUT_SECONDS = int(os.getenv("INDIC_TRANS_TIMEOUT_SECONDS") or 120)


class TranslateRequest(BaseModel):
    text: str
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


class Segment(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    text: str


class TranslateSegmentsRequest(BaseModel):
    segments: List[Segment]
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


class _ModelBundle:
    def __init__(self, model_name: str):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"Required dependency missing: {_IMPORT_ERROR}")
        kwargs: Dict[str, Any] = {}
        if CACHE_DIR:
            kwargs["cache_dir"] = CACHE_DIR
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            **kwargs,
        )
        device = "cpu"
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        self.device = device
        if torch is not None:
            self.model.to(self.device)
        self.lock = threading.Lock()


_MODEL_CACHE: Dict[str, _ModelBundle] = {}
_REQUEST_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
_LAST_CLEANUP_TIME = time.time()


def _normalize_lang(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    raw = str(code).strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower in LANG_MAP:
        return LANG_MAP[lower]
    # accept IndicTrans2 codes directly
    if "_" in raw:
        return raw
    return raw


def _is_indic(code: Optional[str]) -> bool:
    if not code:
        return False
    iso = str(code).strip().lower()
    return iso in LANG_MAP and iso != "en"


def _get_memory_usage() -> float:
    """Get current memory usage percentage."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        return (mem_info.rss / system_mem.total) * 100
    except Exception:
        return 0.0


def _cleanup_memory() -> None:
    """Force garbage collection and clear PyTorch cache if needed."""
    global _LAST_CLEANUP_TIME
    current_time = time.time()
    # Only cleanup if at least 60 seconds have passed
    if current_time - _LAST_CLEANUP_TIME < 60:
        return
    
    mem_usage = _get_memory_usage()
    if mem_usage > MEMORY_LIMIT_PCT:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _LAST_CLEANUP_TIME = current_time


def _resolve_model_name(src_iso: str, tgt_iso: str) -> str:
    if src_iso == "en" and tgt_iso != "en":
        return MODEL_EN_INDIC or MODEL_DEFAULT
    if src_iso != "en" and tgt_iso == "en":
        return MODEL_INDIC_EN or MODEL_DEFAULT
    if src_iso != "en" and tgt_iso != "en":
        return MODEL_INDIC_INDIC or MODEL_DEFAULT
    return MODEL_DEFAULT or MODEL_EN_INDIC


def _get_bundle(model_name: str) -> _ModelBundle:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = _ModelBundle(model_name)
    return _MODEL_CACHE[model_name]


def _get_forced_bos_id(tokenizer: Any, tgt_code: str) -> Optional[int]:
    if hasattr(tokenizer, "get_lang_id"):
        try:
            return tokenizer.get_lang_id(tgt_code)
        except Exception:
            return None
    if hasattr(tokenizer, "lang_code_to_id"):
        try:
            return tokenizer.lang_code_to_id.get(tgt_code)
        except Exception:
            return None
    return None


def _translate_batch(texts: List[str], src_code: str, tgt_code: str, model_name: str) -> List[str]:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"Required dependency missing: {_IMPORT_ERROR}")
    
    # Check memory before processing
    mem_usage = _get_memory_usage()
    if mem_usage > MEMORY_LIMIT_PCT:
        _cleanup_memory()
        # Check again after cleanup
        mem_usage = _get_memory_usage()
        if mem_usage > 95.0:
            raise HTTPException(status_code=503, detail=f"Server memory critical: {mem_usage:.1f}%")
    
    # Truncate texts to prevent OOM
    texts = [t[:MAX_TEXT_LENGTH] for t in texts]
    
    bundle = _get_bundle(model_name)
    tokenizer = bundle.tokenizer
    model = bundle.model

    results: List[str] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [f"{src_code} {tgt_code} {t}" for t in texts[i : i + BATCH_SIZE]]
        with bundle.lock:
            try:
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                )
                if torch is not None:
                    encoded = {k: v.to(bundle.device) for k, v in encoded.items()}
                forced_bos_token_id = _get_forced_bos_id(tokenizer, tgt_code)
                gen_kwargs = {"max_length": MAX_TOKENS}
                if forced_bos_token_id is not None:
                    gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
                output = model.generate(**encoded, **gen_kwargs)
                decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
                results.extend([str(x).strip() for x in decoded])
            finally:
                # Clean up tensors after each batch
                del encoded
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Periodic cleanup between batches
        if (i // BATCH_SIZE) % 5 == 0:
            _cleanup_memory()
    
    return results


@app.get("/health")
def health() -> Dict[str, Any]:
    mem_usage = _get_memory_usage()
    system_mem = psutil.virtual_memory()
    detail = {
        "ok": _IMPORT_ERROR is None and mem_usage < 95.0,
        "device": "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        "memory_usage_pct": round(mem_usage, 2),
        "memory_available_gb": round(system_mem.available / (1024**3), 2),
        "models_loaded": len(_MODEL_CACHE),
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "models": {
            "en_indic": MODEL_EN_INDIC,
            "indic_en": MODEL_INDIC_EN,
            "indic_indic": MODEL_INDIC_INDIC,
            "default": MODEL_DEFAULT,
        },
    }
    if _IMPORT_ERROR is not None:
        detail["error"] = str(_IMPORT_ERROR)
    if mem_usage > MEMORY_LIMIT_PCT:
        detail["warning"] = "High memory usage"
    return detail


@app.get("/languages")
def languages() -> Dict[str, Any]:
    langs = [{"code": code, "name": name} for code, name in LANG_NAMES.items()]
    return {"ok": True, "languages": langs}


@app.post("/translate")
def translate(req: TranslateRequest) -> Dict[str, Any]:
    text = (req.text or "").strip()
    if not text:
        return {"translated_text": ""}

    src_iso = (req.source_lang or "").strip().lower() or "en"
    tgt_iso = (req.target_lang or "").strip().lower()
    if not tgt_iso:
        raise HTTPException(status_code=400, detail="target_lang is required")

    if src_iso == tgt_iso:
        return {"translated_text": text}

    model_name = _resolve_model_name(src_iso, tgt_iso)
    if not model_name:
        raise HTTPException(status_code=500, detail="IndicTrans2 model not configured")

    src_code = _normalize_lang(src_iso)
    tgt_code = _normalize_lang(tgt_iso)
    if not src_code or not tgt_code:
        raise HTTPException(status_code=400, detail="Unsupported language code")

    # Limit concurrent requests
    acquired = _REQUEST_SEMAPHORE.acquire(timeout=REQUEST_TIMEOUT_SECONDS)
    if not acquired:
        raise HTTPException(status_code=503, detail="Service busy, please retry")
    
    try:
        translated = _translate_batch([text], src_code, tgt_code, model_name)[0]
        return {"translated_text": translated}
    finally:
        _REQUEST_SEMAPHORE.release()


@app.post("/translate/segments")
def translate_segments(req: TranslateSegmentsRequest) -> Dict[str, Any]:
    src_iso = (req.source_lang or "").strip().lower() or "en"
    tgt_iso = (req.target_lang or "").strip().lower()
    if not tgt_iso:
        raise HTTPException(status_code=400, detail="target_lang is required")

    if src_iso == tgt_iso:
        return {"segments": [seg.dict() for seg in req.segments]}

    model_name = _resolve_model_name(src_iso, tgt_iso)
    if not model_name:
        raise HTTPException(status_code=500, detail="IndicTrans2 model not configured")

    src_code = _normalize_lang(src_iso)
    tgt_code = _normalize_lang(tgt_iso)
    if not src_code or not tgt_code:
        raise HTTPException(status_code=400, detail="Unsupported language code")

    # Limit concurrent requests
    acquired = _REQUEST_SEMAPHORE.acquire(timeout=REQUEST_TIMEOUT_SECONDS)
    if not acquired:
        raise HTTPException(status_code=503, detail="Service busy, please retry")
    
    try:
        texts = [str(seg.text or "").strip() for seg in req.segments]
        translated = _translate_batch(texts, src_code, tgt_code, model_name)

        out = []
        for seg, text in zip(req.segments, translated):
            payload = seg.dict()
            payload["translated_text"] = text
            out.append(payload)
        return {"segments": out}
    finally:
        _REQUEST_SEMAPHORE.release()


@app.on_event("startup")
def _warmup() -> None:
    warm = (os.getenv("INDIC_TRANS_WARMUP") or "true").strip().lower() in {"1", "true", "yes"}
    if not warm:
        return
    try:
        _ = _resolve_model_name("en", "hi")
        _translate_batch(["hello"], _normalize_lang("en"), _normalize_lang("hi"), _resolve_model_name("en", "hi"))
    except Exception:
        pass
