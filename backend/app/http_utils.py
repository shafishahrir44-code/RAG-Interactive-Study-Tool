"""Shared HTTP helper that maps transport/API failures to application errors."""

from __future__ import annotations

from typing import Any

import httpx

from app.errors import AppError

TIMEOUT_STATUS = 408  # recorded on errors caused by a client-side timeout (no HTTP response)


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    error_cls: type[AppError],
    service_name: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        error = error_cls(f"{service_name} timed out.", detail=f"POST {url}: {exc!r}")
        error.upstream_status = TIMEOUT_STATUS
        raise error from exc
    except httpx.HTTPError as exc:
        raise error_cls(
            f"Could not reach {service_name}. Is it running and is the URL correct?",
            detail=f"POST {url}: {exc!r}",
        ) from exc
    if response.status_code >= 400:
        error = error_cls(
            f"{service_name} returned an error (HTTP {response.status_code}).",
            detail=f"POST {url}: {response.text[:500]}",
        )
        error.upstream_status = response.status_code
        raise error
    try:
        return response.json()
    except ValueError as exc:
        raise error_cls(f"{service_name} returned invalid JSON.", detail=response.text[:500]) from exc


def bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}
