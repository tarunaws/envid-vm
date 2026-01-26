from __future__ import annotations

import io
import os
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

try:
    from docx import Document as DocxDocument  # type: ignore
except Exception:
    DocxDocument = None


def _extract_text_from_bytes(file_content: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("PDF support not installed")
        reader = PdfReader(io.BytesIO(file_content))
        parts = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts).strip()
    if name.endswith(".docx"):
        if DocxDocument is None:
            raise RuntimeError("DOCX support not installed")
        doc = DocxDocument(io.BytesIO(file_content))
        return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()
    return file_content.decode("utf-8", errors="replace").strip()


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True}), 200


@app.post("/extract-text")
def extract_text() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400
    f = request.files["file"]
    filename = f.filename or "document"
    raw = f.read()
    if not raw:
        return jsonify({"error": "empty file"}), 400
    try:
        text = _extract_text_from_bytes(raw, filename)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"text": text, "filename": filename}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5097")))
