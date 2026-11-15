from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from inference.plugins.llm.tool_calling.config import WebSearchToolConfig
from inference.plugins.llm.tool_calling.safety import sanitize_untrusted_text

logger = logging.getLogger(__name__)

WEB_SEARCH_OPENAI_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web for up-to-date information. "
            "Use when the user asks for recent news, current facts, or citations from the web."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise search query in the user's language when possible.",
                },
            },
            "required": ["query"],
        },
    },
}


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("search provider returned non-object JSON")
    return parsed


async def run_web_search(config: WebSearchToolConfig, arguments: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    query = str(arguments.get("query", "") or "").strip()
    if not query:
        return {"ok": False, "error": "missing_query", "results": []}
    if not config.provider_ready():
        return {
            "ok": False,
            "error": "web_search_disabled",
            "message": "Web search is not configured (missing API key or disabled in config).",
            "results": [],
        }

    try:
        if config.provider == "tavily":
            payload = await asyncio.to_thread(
                _tavily_search,
                config,
                query,
            )
        elif config.provider == "brave":
            payload = await asyncio.to_thread(
                _brave_search,
                config,
                query,
            )
        else:
            return {
                "ok": False,
                "error": "unknown_provider",
                "provider": config.provider,
                "results": [],
            }
    except urllib.error.HTTPError as exc:
        logger.warning("web_search HTTP error provider=%s code=%s", config.provider, exc.code)
        return {
            "ok": False,
            "error": "http_error",
            "status": exc.code,
            "results": [],
        }
    except TimeoutError:
        return {"ok": False, "error": "timeout", "results": []}
    except Exception as exc:  # noqa: BLE001 — surface structured tool error
        logger.warning("web_search failed provider=%s: %s", config.provider, exc)
        return {"ok": False, "error": "search_failed", "message": str(exc), "results": []}

    return payload


def _tavily_search(config: WebSearchToolConfig, query: str) -> dict[str, Any]:
    url = config.base_url or "https://api.tavily.com/search"
    raw = _http_post_json(
        url,
        {
            "api_key": config.api_key,
            "query": query,
            "search_depth": config.search_depth,
            "max_results": config.max_results,
            "include_answer": False,
        },
        headers={"Content-Type": "application/json"},
        timeout_s=config.timeout_s,
    )
    items = []
    for hit in raw.get("results") or []:
        if not isinstance(hit, dict):
            continue
        url_hit = str(hit.get("url", "") or "")
        title = sanitize_untrusted_text(str(hit.get("title", "") or ""), max_chars=200)
        snippet = sanitize_untrusted_text(str(hit.get("content", "") or ""), max_chars=config.max_snippet_chars)
        if url_hit:
            items.append({"url": url_hit, "title": title, "snippet": snippet})
    return {"ok": True, "provider": "tavily", "query": query, "results": items}


def _brave_search(config: WebSearchToolConfig, query: str) -> dict[str, Any]:
    url = config.base_url or "https://api.search.brave.com/res/v1/web/search"
    import urllib.parse

    qs = urllib.parse.urlencode({"q": query, "count": config.max_results})
    req = urllib.request.Request(
        f"{url}?{qs}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": config.api_key,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    raw = json.loads(body)
    web = (raw.get("web") or {}) if isinstance(raw, dict) else {}
    hits = web.get("results") or []
    items = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        url_hit = str(hit.get("url", "") or "")
        title = sanitize_untrusted_text(str(hit.get("title", "") or ""), max_chars=200)
        snippet = sanitize_untrusted_text(str(hit.get("description", "") or ""), max_chars=config.max_snippet_chars)
        if url_hit:
            items.append({"url": url_hit, "title": title, "snippet": snippet})
    return {"ok": True, "provider": "brave", "query": query, "results": items}