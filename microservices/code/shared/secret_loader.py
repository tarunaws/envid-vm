"""Helpers to hydrate environment variables from AWS Secrets Manager."""
from __future__ import annotations

import json
import logging
import os
from typing import Iterable, Mapping

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None
    BotoCoreError = ClientError = Exception

LOGGER = logging.getLogger(__name__)


def load_aws_secret_into_env(
    secret_id: str | None,
    *,
    region: str | None = None,
    required_keys: Iterable[str] | None = None,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> Mapping[str, str]:
    """Fetch ``secret_id`` from AWS Secrets Manager and project it into ``os.environ``."""
    if not secret_id or not boto3:
        return {}

    log = logger or LOGGER
    session = boto3.session.Session()
    try:
        client = session.client("secretsmanager", region_name=region)
    except Exception as exc:  # pragma: no cover - boto3 misconfiguration
        log.warning("Unable to create secretsmanager client: %s", exc)
        return {}

    try:
        response = client.get_secret_value(SecretId=secret_id)
    except (BotoCoreError, ClientError) as exc:
        log.warning("Failed to fetch secret %s: %s", secret_id, exc)
        return {}

    secret_string = response.get("SecretString") or ""
    payload: dict[str, str] = {}
    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        log.warning("Secret %s is not valid JSON: %s", secret_id, exc)
        return {}

    hydrated: dict[str, str] = {}
    for key, value in payload.items():
        if required_keys and key not in required_keys:
            continue
        if not overwrite and key in os.environ:
            continue
        os.environ[key] = str(value)
        hydrated[key] = str(value)

    if hydrated:
        log.info("Loaded %d secrets from %s", len(hydrated), secret_id)
    else:
        log.info("Secret %s did not produce new environment entries", secret_id)
    return hydrated
