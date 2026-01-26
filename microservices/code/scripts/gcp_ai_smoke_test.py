"""Cloud AI/API smoke tests for Envid Metadata.

Runs lightweight calls to confirm the service account can reach required APIs:
- GCS (list rawVideo/)
- Translate v3
- Speech-to-Text v1
- Natural Language API
- Video Intelligence
- Transcoder
- Vertex AI Gemini (generateContent)

Usage (from repo root):
    # Strict (default): fail if any check fails
    code/.venv/bin/python3 code/scripts/gcp_ai_smoke_test.py

    # Non-strict: only fail if required checks fail (currently GCS)
    code/.venv/bin/python3 code/scripts/gcp_ai_smoke_test.py --non-strict

    # Include API-key based checks (Translate v2 + Gemini AI Studio)
    # NOTE: These validate API enablement + key validity, not service-account IAM.
    code/.venv/bin/python3 code/scripts/gcp_ai_smoke_test.py --with-apikey
"""

from __future__ import annotations

import json
import os
import sys
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass
class CheckResult:
    ok: bool
    detail: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _record(out: Dict[str, Any], name: str, res: CheckResult) -> None:
    payload: Dict[str, Any] = {"ok": bool(res.ok)}
    if res.detail is not None:
        payload["detail"] = res.detail
    if res.error is not None:
        payload["error"] = res.error
    out.setdefault("checks", {})[name] = payload


def _exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def _summarize_exception(exc: Exception) -> Dict[str, Any]:
    """Best-effort extraction of reason/permission/service from common GCP errors."""

    msg = str(exc)
    upper = msg.upper()
    summary: Dict[str, Any] = {}

    if "SERVICE_DISABLED" in upper:
        summary["reason"] = "SERVICE_DISABLED"
    elif "IAM_PERMISSION_DENIED" in upper:
        summary["reason"] = "IAM_PERMISSION_DENIED"
    elif "PERMISSION_DENIED" in upper:
        summary["reason"] = "PERMISSION_DENIED"

    # Common patterns in error strings
    m = re.search(r"Permission '([^']+)' denied", msg)
    if m:
        summary["permission"] = m.group(1)

    m = re.search(r"value: \"([^\"]+\.googleapis\.com)\"", msg)
    if m:
        summary["service"] = m.group(1)

    m = re.search(r"activationUrl\"\s*\n\s*value: \"([^\"]+)\"", msg)
    if m:
        summary["activationUrl"] = m.group(1)

    return summary


def _summarize_http_error(body_text: str) -> Dict[str, Any]:
    """Extract useful fields from Google REST error bodies."""

    try:
        data = json.loads(body_text)
    except Exception:
        return {}

    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return {}

    summary: Dict[str, Any] = {}
    if isinstance(err.get("status"), str):
        summary["status"] = err.get("status")
    if isinstance(err.get("message"), str):
        summary["message"] = err.get("message")

    # Try to locate ErrorInfo.reason + permission
    details = err.get("details")
    if isinstance(details, list):
        for d in details:
            if not isinstance(d, dict):
                continue
            if d.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo":
                meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
                if isinstance(d.get("reason"), str):
                    summary["reason"] = d.get("reason")
                if isinstance(meta.get("permission"), str):
                    summary["permission"] = meta.get("permission")
                if isinstance(meta.get("resource"), str):
                    summary["resource"] = meta.get("resource")
                break
    return summary


def _is_expected_nonfatal(exc: Exception) -> tuple[bool, str]:
    """Return (is_expected, reason) for errors that are common in dev environments."""

    msg = str(exc)
    upper = msg.upper()

    if isinstance(exc, ModuleNotFoundError):
        return True, "missing_dependency"

    if "SERVICE_DISABLED" in upper or "HAS NOT BEEN USED" in upper or "IS DISABLED" in upper:
        return True, "api_disabled"

    if "IAM_PERMISSION_DENIED" in upper or "PERMISSION_DENIED" in upper or "PERMISSION '" in upper:
        return True, "iam_denied"

    return False, ""


def _record_exception(out: Dict[str, Any], name: str, exc: Exception, *, strict: bool) -> None:
    expected, reason = _is_expected_nonfatal(exc)
    if expected and not strict:
        _record(
            out,
            name,
            CheckResult(True, detail={"skipped": True, "reason": reason, "summary": _summarize_exception(exc)}, error=_exc(exc)),
        )
        return
    _record(out, name, CheckResult(False, detail={"summary": _summarize_exception(exc)}, error=_exc(exc)))


def main() -> int:
    args = set(sys.argv[1:])
    strict = True
    if "--non-strict" in args:
        strict = False

    # API key checks are optional by default because they don't validate SA IAM.
    with_apikey_checks = "--with-apikey" in args

    # Load local env file if present (keeps this script runnable without shell `source`).
    if load_dotenv is not None:
        repo_root = Path(__file__).resolve().parents[2]
        for rel in (".env", ".env.local"):
            env_path = repo_root / rel
            if env_path.exists():
                load_dotenv(env_path, override=False)
        for rel in (".env.multimodal.local", ".env.multimodal.secrets.local", ".env.multimodal.vm.local"):
            env_path = repo_root / "backend" / rel
            if env_path.exists():
                load_dotenv(env_path, override=False)

    project = _env("GCP_PROJECT_ID")
    location = _env("GCP_LOCATION", "us-east1")
    bucket = _env("GCP_GCS_BUCKET")
    creds_path = _env("GOOGLE_APPLICATION_CREDENTIALS")
    api_key = _env("GCP_API_KEY")

    out: Dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "strict": strict,
        "env": {
            "GCP_PROJECT_ID": project,
            "GCP_LOCATION": location,
            "GCP_GCS_BUCKET": bucket,
            "GOOGLE_APPLICATION_CREDENTIALS_present": bool(creds_path),
            "GOOGLE_APPLICATION_CREDENTIALS_path": creds_path,
            "GOOGLE_APPLICATION_CREDENTIALS_exists": bool(creds_path and Path(creds_path).exists()),
            "GCP_API_KEY_present": bool(api_key),
        },
        "checks": {},
    }

    # GCS list
    try:
        from google.cloud import storage  # type: ignore

        c = storage.Client()
        blobs = list(c.list_blobs(bucket, prefix="rawVideo/", max_results=10))
        _record(
            out,
            "gcs.list_rawVideo",
            CheckResult(True, {"count": len(blobs), "sample": [b.name for b in blobs]}),
        )
    except Exception as e:
        _record_exception(out, "gcs.list_rawVideo", e, strict=strict)

    # Translate v3
    try:
        from google.cloud import translate_v3  # type: ignore

        client = translate_v3.TranslationServiceClient()
        parent = f"projects/{project}/locations/global"
        resp = client.translate_text(
            request={
                "parent": parent,
                "contents": ["hello world"],
                "mime_type": "text/plain",
                "target_language_code": "hi",
            }
        )
        translated = resp.translations[0].translated_text if resp.translations else None
        _record(out, "translate.translate_text", CheckResult(True, {"translated": translated}))
    except Exception as e:
        _record_exception(out, "translate.translate_text", e, strict=strict)

    # Translate via API key (Translate v2 REST)
    # NOTE: This validates API enablement + key validity, but does not validate service-account IAM.
    if api_key and with_apikey_checks:
        try:
            import requests  # type: ignore

            url = "https://translation.googleapis.com/language/translate/v2"
            r = requests.post(
                url,
                params={"key": api_key},
                json={"q": "hello world", "target": "hi", "format": "text"},
                timeout=20,
            )
            if r.status_code >= 400:
                body = (r.text or "")
                summary = _summarize_http_error(body)
                _record(
                    out,
                    "translate_v2.apikey.translate",
                    CheckResult(
                        False if strict else True,
                        detail={
                            "status": r.status_code,
                            "body": body[:500],
                            "summary": summary,
                            "skipped": (not strict),
                        },
                        error=(f"HTTP {r.status_code}" if strict else "HTTP error treated as skipped (non-strict)"),
                    ),
                )
            else:
                data = r.json() if r.content else {}
                translated = None
                try:
                    translated = data["data"]["translations"][0].get("translatedText")
                except Exception:
                    translated = None
                _record(
                    out,
                    "translate_v2.apikey.translate",
                    CheckResult(True, {"status": r.status_code, "translated": translated}),
                )
        except Exception as e:
            _record_exception(out, "translate_v2.apikey.translate", e, strict=strict)
    elif api_key and not with_apikey_checks:
        _record(
            out,
            "translate_v2.apikey.translate",
            CheckResult(True, detail={"skipped": True, "reason": "apikey_checks_disabled"}),
        )

    # Natural Language API (analyze entities)
    try:
        from google.cloud import language_v1  # type: ignore

        client = language_v1.LanguageServiceClient()
        doc = language_v1.Document(content="Paris is a city.", type_=language_v1.Document.Type.PLAIN_TEXT)
        resp = client.analyze_entities(request={"document": doc})
        entities = []
        for e in (resp.entities or [])[:5]:
            entities.append({"name": getattr(e, "name", None), "type": str(getattr(e, "type_", ""))})
        _record(out, "language.analyze_entities", CheckResult(True, {"entities": entities}))
    except Exception as e:
        _record_exception(out, "language.analyze_entities", e, strict=strict)

    # Speech-to-Text v1 (make a tiny request expected to fail with 400 if auth+API are OK)
    try:
        from google.cloud import speech_v1  # type: ignore

        client = speech_v1.SpeechClient()
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        audio = speech_v1.RecognitionAudio(content=b"")
        try:
            client.recognize(config=config, audio=audio)
            # Unexpected, but still indicates API access works.
            _record(out, "speech_v1.recognize", CheckResult(True, {"note": "unexpected_success"}))
        except Exception as inner:
            # We expect InvalidArgument for empty audio if IAM/API are fine.
            expected, reason = _is_expected_nonfatal(inner)
            if "INVALID_ARGUMENT" in str(inner).upper() or "400" in str(inner):
                _record(out, "speech_v1.recognize", CheckResult(True, {"expected_error": _exc(inner)}))
            elif expected and not strict:
                _record(
                    out,
                    "speech_v1.recognize",
                    CheckResult(True, detail={"skipped": True, "reason": reason, "summary": _summarize_exception(inner)}, error=_exc(inner)),
                )
            else:
                raise
    except Exception as e:
        _record_exception(out, "speech_v1.recognize", e, strict=strict)

    # Video Intelligence (list operations)
    try:
        from google.cloud import videointelligence  # type: ignore

        client = videointelligence.VideoIntelligenceServiceClient()
        parent = f"projects/{project}/locations/{location}"
        # VideoIntelligenceServiceClient doesn't expose list_operations() directly; use its operations client.
        ops_client = getattr(client.transport, "operations_client", None)
        if ops_client is None or not hasattr(ops_client, "list_operations"):
            raise RuntimeError("Video Intelligence operations client missing list_operations")

        # OperationsClient.list_operations signature differs from GAPIC request-style.
        ops = ops_client.list_operations(name=parent, filter_=None)
        first = None
        for op in ops:
            first = getattr(op, "name", None)
            break
        _record(out, "videointelligence.list_operations", CheckResult(True, {"first": first}))
    except Exception as e:
        _record_exception(out, "videointelligence.list_operations", e, strict=strict)

    # Transcoder (list jobs)
    try:
        from google.cloud.video import transcoder_v1  # type: ignore

        client = transcoder_v1.TranscoderServiceClient()
        parent = f"projects/{project}/locations/{location}"
        jobs = client.list_jobs(request={"parent": parent, "page_size": 1})
        first = None
        for j in jobs:
            first = getattr(j, "name", None)
            break
        _record(out, "transcoder.list_jobs", CheckResult(True, {"first": first}))
    except Exception as e:
        _record_exception(out, "transcoder.list_jobs", e, strict=strict)

    # Vertex AI Gemini (generateContent)
    try:
        import google.auth  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore
        import requests  # type: ignore

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request())
        token = creds.token

        model = _env("GCP_GEMINI_MODEL", "gemini-1.5-flash")
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Return the word OK."}]}],
            "generationConfig": {"temperature": 0},
        }
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        if r.status_code >= 400:
            body = (r.text or "")
            summary = _summarize_http_error(body)
            _record(
                out,
                "vertexai.gemini.generateContent",
                CheckResult(
                    False if strict else True,
                    detail={
                        "status": r.status_code,
                        "body": body[:500],
                        "summary": summary,
                        "skipped": (not strict),
                    },
                    error=(f"HTTP {r.status_code}" if strict else "HTTP error treated as skipped (non-strict)"),
                ),
            )
        else:
            data = r.json() if r.content else {}
            text = None
            try:
                text = data["candidates"][0]["content"]["parts"][0].get("text")
            except Exception:
                text = None
            _record(
                out,
                "vertexai.gemini.generateContent",
                CheckResult(True, {"status": r.status_code, "text": text}),
            )
    except Exception as e:
        _record_exception(out, "vertexai.gemini.generateContent", e, strict=strict)

    # Gemini via API key (Google AI Studio / Generative Language API)
    # NOTE: This is *not* Vertex AI. It validates whether the provided API key can reach Gemini.
    if api_key and with_apikey_checks:
        try:
            import requests  # type: ignore

            model = _env("GEMINI_API_STUDIO_MODEL", "gemini-1.5-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": "Return the word OK."}]}],
                "generationConfig": {"temperature": 0},
            }
            r = requests.post(
                url,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=20,
            )
            if r.status_code >= 400:
                body = (r.text or "")
                summary = _summarize_http_error(body)
                _record(
                    out,
                    "gemini_ai_studio.apikey.generateContent",
                    CheckResult(
                        False if strict else True,
                        detail={
                            "status": r.status_code,
                            "body": body[:500],
                            "summary": summary,
                            "skipped": (not strict),
                        },
                        error=(f"HTTP {r.status_code}" if strict else "HTTP error treated as skipped (non-strict)"),
                    ),
                )
            else:
                data = r.json() if r.content else {}
                text = None
                try:
                    text = data["candidates"][0]["content"]["parts"][0].get("text")
                except Exception:
                    text = None
                _record(
                    out,
                    "gemini_ai_studio.apikey.generateContent",
                    CheckResult(True, {"status": r.status_code, "text": text}),
                )
        except Exception as e:
            _record_exception(out, "gemini_ai_studio.apikey.generateContent", e, strict=strict)
    elif api_key and not with_apikey_checks:
        _record(
            out,
            "gemini_ai_studio.apikey.generateContent",
            CheckResult(True, detail={"skipped": True, "reason": "apikey_checks_disabled"}),
        )

    required = ["gcs.list_rawVideo"]
    failed_required = []
    for name in required:
        chk = (out.get("checks") or {}).get(name) or {}
        if not chk.get("ok"):
            failed_required.append(name)

    if strict:
        any_failed = [k for k, v in (out.get("checks") or {}).items() if not (v or {}).get("ok")]
        out["exit_reason"] = {"mode": "strict", "failed": any_failed}
        print(json.dumps(out, indent=2))
        return 1 if any_failed else 0

    out["exit_reason"] = {"mode": "non_strict", "failed_required": failed_required}
    print(json.dumps(out, indent=2))
    return 1 if failed_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
