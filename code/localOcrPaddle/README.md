# local-ocr-paddle

Dockerized PaddleOCR service (Python 3.11) used by the main Envid Multimodal backend (Python 3.14+) to perform on-screen text OCR without requiring `paddlepaddle` in the main venv.

## Endpoints

- `GET /health`
- `POST /ocr/frames`

Request:
```json
{
  "lang": "en",
  "frame_len": 2.0,
  "frames": [
    {"time": 12.3, "image_b64": "...", "image_mime": "image/jpeg"}
  ]
}
```

Response:
```json
{
  "text": [
    {"text": "Hello", "segments": [{"start": 12.3, "end": 14.3, "confidence": 0.92}]}
  ]
}
```

## Notes

- First run may download OCR model assets.
- If `paddlepaddle` install fails for your CPU/arch, adjust the pinned version in `requirements.txt`.
