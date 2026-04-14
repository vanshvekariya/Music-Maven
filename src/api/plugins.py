"""Subgroup plugin registry and HTTP forward (Phase E)."""

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from ..config.settings import get_settings

router = APIRouter(prefix="/plugins", tags=["Plugins"])

# module name -> settings attribute holding full POST URL
PLUGIN_REGISTRY: Dict[str, str] = {
    "tempo": "plugin_tempo_url",
    "artist_classifier": "plugin_artist_classifier_url",
    "lyrics": "plugin_lyrics_url",
}


class PluginForwardBody(BaseModel):
    """Opaque JSON forwarded to the subgroup service (contract is team-specific)."""

    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON body sent as-is to the configured plugin URL",
    )


def plugins_configured_map() -> Dict[str, bool]:
    """Which subgroup modules have a non-empty URL in settings (for /system/info)."""
    s = get_settings()
    out: Dict[str, bool] = {}
    for name, attr in PLUGIN_REGISTRY.items():
        raw = getattr(s, attr, None)
        out[name] = bool(raw and str(raw).strip())
    return out


def _url_for_module(name: str) -> Optional[str]:
    settings = get_settings()
    attr = PLUGIN_REGISTRY.get(name)
    if not attr:
        return None
    url = getattr(settings, attr, None)
    if url is None or not str(url).strip():
        return None
    return str(url).strip()


@router.get("/status")
async def plugins_status() -> Dict[str, Any]:
    """List subgroup modules and whether each has a configured forward URL."""
    return {
        "modules": [
            {
                "name": name,
                "configured": _url_for_module(name) is not None,
            }
            for name in PLUGIN_REGISTRY
        ],
        "forward_contract": (
            "POST /plugins/{module}/forward with body {\"payload\": {...}}; "
            "server forwards payload as JSON to the URL in .env for that module."
        ),
    }


@router.post("/{module}/forward")
async def plugins_forward(module: str, body: PluginForwardBody) -> Dict[str, Any]:
    """
    Proxy JSON to a subgroup HTTP service when PLUGIN_*_URL is set.

    The subgroup defines the payload shape; Music Maven does not validate it beyond JSON.
    """
    if module not in PLUGIN_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown plugin module '{module}'. Valid: {list(PLUGIN_REGISTRY)}",
        )

    url = _url_for_module(module)
    if not url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Plugin '{module}' is not configured. "
                f"Set {PLUGIN_REGISTRY[module].upper()} in .env to the team's POST endpoint."
            ),
        )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=body.payload)
    except httpx.RequestError as e:
        logger.error(f"Plugin forward failed for {module}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach plugin service: {e!s}",
        ) from e

    ct = response.headers.get("content-type", "")
    if "application/json" in ct:
        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text}
    else:
        data = {"raw": response.text}

    return {
        "module": module,
        "upstream_status": response.status_code,
        "data": data,
    }
