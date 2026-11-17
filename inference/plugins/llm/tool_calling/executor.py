from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from inference.plugins.llm.tool_calling.config import ToolCallingSettings
from inference.plugins.llm.tool_calling.safety import sanitize_untrusted_text
from inference.plugins.llm.tool_calling.builtins import web_search

logger = logging.getLogger(__name__)

_ALLOWED_TOOLS = frozenset({"web_search"})


class ToolExecutor:
    def __init__(self, settings: ToolCallingSettings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_tools)

    def openai_tool_definitions(self) -> list[dict[str, Any]]:
        defs: list[dict[str, Any]] = []
        if self.settings.web_search.provider_ready():
            defs.append(web_search.WEB_SEARCH_OPENAI_SCHEMA)
        return defs

    async def execute_many(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not calls:
            return []
        tasks = [self._execute_one(call) for call in calls]
        return await asyncio.gather(*tasks)

    async def _execute_one(self, call: dict[str, Any]) -> dict[str, Any]:
        name = str(call.get("name", "") or "")
        call_id = str(call.get("id", "") or "")
        if name not in _ALLOWED_TOOLS:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {"ok": False, "error": "tool_not_allowed", "name": name},
                    ensure_ascii=False,
                ),
            }
        args_raw = call.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(args_raw, dict):
            arguments = args_raw
        else:
            arguments = {}

        async with self._semaphore:
            if name == "web_search":
                result = await web_search.run_web_search(self.settings.web_search, arguments)
            else:
                result = {"ok": False, "error": "unimplemented"}

        content = json.dumps(result, ensure_ascii=False)
        content = sanitize_untrusted_text(content, max_chars=self.settings.max_tool_result_chars)
        return {"role": "tool", "tool_call_id": call_id, "content": content}