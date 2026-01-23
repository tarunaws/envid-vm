# Local LibreTranslate (Docker Compose)

Production-ready local LibreTranslate setup with a reverse proxy and cached models.

## Start/Stop

```bash
cp .env.sample .env
make up
```

Stop:

```bash
make down
```

Logs:

```bash
make logs
```

Health check:

```bash
make health
```

## Pre-download models

This caches Argos models into the named volume for fast startup:

```bash
make models
```

## Endpoints

- Direct LibreTranslate: http://localhost:5000
- Proxy: http://localhost:8080

Available endpoints:
- `/translate`
- `/languages`
- `/detect`

## Example curl

Translate:

```bash
curl -s http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LT_AUTH_TOKEN" \
  -d '{"q":"Hello","source":"en","target":"hi"}'
```

Detect:

```bash
curl -s http://localhost:8080/detect \
  -H "Content-Type: application/json" \
  -d '{"q":"नमस्ते दुनिया"}'
```

List languages:

```bash
curl -s http://localhost:8080/languages
```

## Troubleshooting

- **Model not found**: run `make models` to pre-download the Argos models.
- **Slow startup**: reduce `LT_WORKERS` or confirm models are cached.
- **Memory pressure**: lower `LT_WORKERS`/`LT_BATCH_SIZE` and increase `LT_MEM_LIMIT`.
- **Auth errors**: if `LT_AUTH_TOKEN` is set, pass `Authorization: Bearer <token>`.

## Clients

See example clients:
- [example/clients/python_client.py](example/clients/python_client.py)
- [example/clients/node_client.js](example/clients/node_client.js)
