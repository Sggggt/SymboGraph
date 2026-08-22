from __future__ import annotations

import os
import re
from collections.abc import Mapping


_PRIVATE_ENV_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|AUTHORIZATION|"
    r"CREDENTIAL|COOKIE|SESSION|BASE_URL|RESOLVE_IP|DATABASE_URL|REDIS_URL|"
    r"QDRANT_URL|PROXY)(?:_|$)",
    flags=re.IGNORECASE,
)


def sanitized_subprocess_env(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy process environment without forwarding local credentials/endpoints."""

    env = {
        key: value
        for key, value in os.environ.items()
        if _PRIVATE_ENV_NAME.search(key) is None
    }
    if overrides:
        env.update({str(key): str(value) for key, value in overrides.items()})
    return env
