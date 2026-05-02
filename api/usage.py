"""
Hermes Web UI -- Provider usage limits.

Fetches live usage data directly from provider APIs (MiniMax, Z.AI) and
surfaces it to the frontend for the rail icon tooltip.

Usage data is cached per-provider with a 5-minute staleness window to
avoid hammering provider APIs on every page load.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─── Live API URLs ─────────────────────────────────────────────────────────────

_MINIMAX_USAGE_URL = "https://platform.minimax.io/v1/api/openplatform/coding_plan/remains"
_ZAI_USAGE_URL = "https://api.z.ai/api/monitor/usage/model-usage"
_ZAI_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"

# ─── In-memory cache (5-min staleness per provider) ──────────────────────────

_usage_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_get(provider: str) -> Optional[list[dict[str, Any]]]:
    with _cache_lock:
        entry = _usage_cache.get(provider)
        if entry is None:
            return None
        ts, data = entry
        if time.time() - ts > _CACHE_TTL_SECONDS:
            del _usage_cache[provider]
            return None
        return data


def _cache_set(provider: str, data: list[dict[str, Any]]) -> None:
    with _cache_lock:
        _usage_cache[provider] = (time.time(), data)


# ─── Config / key access ──────────────────────────────────────────────────────

def _get_api_key(provider: str) -> Optional[str]:
    """Return the API key for a provider, or None if not configured."""
    env_var_map = {
        "minimax": "MINIMAX_API_KEY",
        "zai": "GLM_API_KEY",
    }
    env_var = env_var_map.get(provider)
    if not env_var:
        return None

    # 1. Environment variable (takes precedence)
    key = os.getenv(env_var)
    if key:
        return key

    # 2. .env file in hermes home
    try:
        from api.profiles import get_active_hermes_home
        home = get_active_hermes_home()
    except Exception:
        home = Path.home()

    env_path = home / ".env"
    if env_path.is_file():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == env_var:
                        return v.strip().strip("\"'")
        except Exception:
            pass

    return None


def _get_minimax_group_id() -> Optional[str]:
    """Return the MiniMax GroupId (stored as api_url in config)."""
    try:
        from api.config import get_config
        cfg = get_config()
        providers = cfg.get("providers", {})
        minimax = providers.get("minimax", {})
        # api_url field stores the GroupId for MiniMax
        group_id = minimax.get("api_url") or minimax.get("group_id")
        if group_id:
            return str(group_id)
    except Exception:
        pass

    # Fallback: MINIMAX_GROUP_ID env var
    return os.getenv("MINIMAX_GROUP_ID")


# ─── MiniMax API ──────────────────────────────────────────────────────────────

def _fetch_minimax_usage() -> Optional[dict[str, Any]]:
    api_key = _get_api_key("minimax")
    group_id = _get_minimax_group_id()
    if not api_key:
        return None
    if not group_id:
        logger.debug("MiniMax usage: no group_id configured")
        return None

    try:
        import urllib.request

        url = f"{_MINIMAX_USAGE_URL}?GroupId={group_id}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        # Response: { model_remains: [{ model_name, current_interval_total_count,
        #                               current_interval_usage_count,
        #                               current_weekly_total_count,
        #                               current_weekly_usage_count,
        #                               end_time, weekly_end_time }] }
        model_remains = data.get("model_remains", [])

        # Find the coding plan entry (covers both "coding-plan-vlm"/"coding-plan-search"
        # and model-specific entries like "MiniMax-M2.5-highspeed")
        coding_plan = None
        for m in model_remains:
            name = m.get("model_name", "")
            if name in ("coding-plan-vlm", "coding-plan-search") or name.startswith("MiniMax-M"):
                coding_plan = m
                break

        if not coding_plan:
            logger.warning("MiniMax usage: no coding plan found in API response")
            return None

        total_hourly = int(coding_plan.get("current_interval_total_count", 0))
        used_hourly = total_hourly - int(coding_plan.get("current_interval_usage_count", 0))
        total_weekly = int(coding_plan.get("current_weekly_total_count", 0))
        used_weekly = total_weekly - int(coding_plan.get("current_weekly_usage_count", 0))

        return {
            "provider": "minimax",
            "limit_5h": total_hourly,
            "used_5h": max(0, used_hourly),
            "limit_7d": total_weekly,
            "used_7d": max(0, used_weekly),
            "reset_5h_ts": coding_plan.get("end_time"),
            "reset_7d_ts": coding_plan.get("weekly_end_time"),
        }
    except Exception as exc:
        logger.warning("Failed to fetch MiniMax usage: %s", exc)
        return None


# ─── Z.AI API ─────────────────────────────────────────────────────────────────

def _fetch_zai_usage() -> Optional[dict[str, Any]]:
    api_key = _get_api_key("zai")
    if not api_key:
        return None

    try:
        import urllib.request
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        week_ago = now - __import__("datetime").timedelta(days=7)
        fmt = lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S").replace(" ", "%20")

        # Fetch usage
        usage_url = f"{_ZAI_USAGE_URL}?startTime={fmt(week_ago)}&endTime={fmt(now)}"
        usage_req = urllib.request.Request(
            usage_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Referer": "https://z.ai/",
            },
        )
        with urllib.request.urlopen(usage_req, timeout=10) as resp:
            usage_data = json.loads(resp.read())

        # modelCallCount is an array of hourly call counts [oldest...newest].
        # Sum the last 5 entries for 5-hour window.
        call_counts = usage_data.get("data", {}).get("modelCallCount", [])
        used_5h = sum(call_counts[-5:]) if len(call_counts) >= 5 else sum(call_counts)
        used_7d = usage_data.get("data", {}).get("totalUsage", {}).get("totalModelCallCount", 0) or 0

        # Fetch quota limits
        limit_5h, limit_7d = 0, 0
        reset_5h_ts, reset_7d_ts = None, None
        try:
            quota_req = urllib.request.Request(
                _ZAI_QUOTA_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "Referer": "https://z.ai/",
                },
            )
            with urllib.request.urlopen(quota_req, timeout=10) as resp:
                quota_data = json.loads(resp.read())

            for limit in quota_data.get("data", {}).get("limits", []):
                lt_type = limit.get("limit_type", "")
                unit = limit.get("unit", 0)
                remaining = int(limit.get("remaining", 0))
                usage_l = int(limit.get("usage", 0))
                total = remaining + usage_l
                if unit == 5 and lt_type == "TIME_LIMIT":
                    limit_5h = total
                    reset_5h_ts = limit.get("next_reset_time")
                elif unit == 6 and lt_type == "TOKENS_LIMIT":
                    limit_7d = total
                    reset_7d_ts = limit.get("next_reset_time")
        except Exception as exc:
            logger.debug("Z.AI quota fetch failed (non-critical): %s", exc)

        return {
            "provider": "zai",
            "limit_5h": limit_5h,
            "used_5h": used_5h,
            "limit_7d": limit_7d,
            "used_7d": used_7d,
            "reset_5h_ts": reset_5h_ts,
            "reset_7d_ts": reset_7d_ts,
        }
    except Exception as exc:
        logger.warning("Failed to fetch Z.AI usage: %s", exc)
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def get_usage_limits() -> list[dict[str, Any]]:
    """
    Return usage limits for MiniMax and Z.AI providers.

    Results are cached for 5 minutes per provider to avoid excessive API calls.

    Each dict contains:
      - provider: str          ("minimax" or "zai")
      - limit_5h: int         (0 if unavailable)
      - used_5h: int
      - limit_7d: int         (0 if unavailable)
      - used_7d: int
      - reset_5h_ts: int | None  (epoch ms, optional)
      - reset_7d_ts: int | None  (epoch ms, optional)
    """
    results: list[dict[str, Any]] = []

    # MiniMax
    cached = _cache_get("minimax")
    if cached is not None:
        results.extend(cached)
    else:
        data = _fetch_minimax_usage()
        if data:
            _cache_set("minimax", [data])
            results.append(data)
        else:
            _cache_set("minimax", [])

    # Z.AI
    cached = _cache_get("zai")
    if cached is not None:
        results.extend(cached)
    else:
        data = _fetch_zai_usage()
        if data:
            _cache_set("zai", [data])
            results.append(data)
        else:
            _cache_set("zai", [])

    return results


def get_enabled_providers() -> list[str]:
    """
    Return provider IDs that have an API key configured.

    Uses the same detection logic as api/providers.py so the rail icons
    match which providers the user has actually configured.
    """
    try:
        from api.providers import get_providers
        providers_data = get_providers()
        return [
            p["id"] for p in providers_data.get("providers", [])
            if p.get("has_key", False)
        ]
    except Exception as exc:
        logger.warning("Could not determine enabled providers: %s", exc)
        return []
