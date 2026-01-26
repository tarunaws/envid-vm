from __future__ import annotations

import os
from typing import Iterable

import requests
from flask import Flask, Response, request

app = Flask(__name__)

BACKEND_URL = (
    os.getenv("BACKEND_URL")
    or "http://backend:5016"
).rstrip("/")


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _filtered_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() != "host"}


def _iter_response(resp: requests.Response) -> Iterable[bytes]:
    for chunk in resp.iter_content(chunk_size=65536):
        if chunk:
            yield chunk


@app.route("/health", methods=["GET"])
def health() -> Response:
    target = f"{BACKEND_URL}/health"
    resp = requests.get(target, timeout=15)
    headers = _filtered_headers(resp.headers)
    return Response(resp.content, status=resp.status_code, headers=headers)


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def proxy(path: str) -> Response:
    target = f"{BACKEND_URL}/{path}" if path else f"{BACKEND_URL}/"
    headers = _filtered_headers(dict(request.headers))
    data = request.get_data(cache=False)
    resp = requests.request(
        method=request.method,
        url=target,
        headers=headers,
        params=request.args,
        data=data,
        stream=True,
        timeout=None,
    )
    response_headers = _filtered_headers(resp.headers)
    return Response(_iter_response(resp), status=resp.status_code, headers=response_headers)


if __name__ == "__main__":
    port = int(os.getenv("PORT") or "5016")
    app.run(host="0.0.0.0", port=port)
