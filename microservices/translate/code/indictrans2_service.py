import os
import time
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

DEVICE_PREF = (os.getenv("INDIC_TRANS_DEVICE") or "auto").strip().lower()
BATCH_SIZE = int(os.getenv("INDIC_TRANS_BATCH_SIZE") or 16)
MAX_TOKENS = int(os.getenv("INDIC_TRANS_MAX_TOKENS") or 512)
CACHE_DIR = (os.getenv("INDIC_TRANS_CACHE_DIR") or "").strip() or None


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        device = "cpu"
        if DEVICE_PREF in {"cuda", "gpu"} and torch is not None and torch.cuda.is_available():
            device = "cuda"
        elif DEVICE_PREF == "auto" and torch is not None and torch.cuda.is_available():
            device = "cuda"
        self.device = device
        if torch is not None:
            self.model.to(self.device)
        self.lock = threading.Lock()


_MODEL_CACHE: Dict[str, _ModelBundle] = {}


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
    bundle = _get_bundle(model_name)
    tokenizer = bundle.tokenizer
    model = bundle.model

    results: List[str] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [f"{src_code} {tgt_code} {t}" for t in texts[i : i + BATCH_SIZE]]
        with bundle.lock:
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
    return results


@app.get("/health")
def health() -> Dict[str, Any]:
    detail = {
        "ok": _IMPORT_ERROR is None,
        "device": "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        "models": {
            "en_indic": MODEL_EN_INDIC,
            "indic_en": MODEL_INDIC_EN,
            "indic_indic": MODEL_INDIC_INDIC,
            "default": MODEL_DEFAULT,
        },
    }
    if _IMPORT_ERROR is not None:
        detail["error"] = str(_IMPORT_ERROR)
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

    translated = _translate_batch([text], src_code, tgt_code, model_name)[0]
    return {"translated_text": translated}


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

    texts = [str(seg.text or "").strip() for seg in req.segments]
    translated = _translate_batch(texts, src_code, tgt_code, model_name)

    out = []
    for seg, text in zip(req.segments, translated):
        payload = seg.dict()
        payload["translated_text"] = text
        out.append(payload)
    return {"segments": out}


@app.on_event("startup")
def _warmup() -> None:
    warm = (os.getenv("INDIC_TRANS_WARMUP") or "false").strip().lower() in {"1", "true", "yes"}
    if not warm:
        return
    try:
        _ = _resolve_model_name("en", "hi")
        _translate_batch(["hello"], _normalize_lang("en"), _normalize_lang("hi"), _resolve_model_name("en", "hi"))
    except Exception:
        pass
