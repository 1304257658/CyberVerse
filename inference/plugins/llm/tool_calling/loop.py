from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

from inference.core.types import LLMResponseChunk
SENTENCE_ENDERS = {"。", "！", "？", ".", "!", "?", "；", ";", "\n"}
from inference.plugins.llm.tool_calling.config import ToolCallingSettings
from inference.plugins.llm.tool_calling.executor import ToolExecutor

logger = logging.getLogger(__name__)


class _ToolCallStreamAccumulator:
    """Accumulate streamed OpenAI tool_call deltas by index."""

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, str]] = {}

    def ingest_delta(self, deltas: list[Any] | None) -> None:
        if not deltas:
            return
        for delta in deltas:
            idx = int(getattr(delta, "index", 0) or 0)
            entry = self._by_index.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if getattr(delta, "id", None):
                entry["id"] = delta.id
            fn = getattr(delta, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    entry["name"] += fn.name
                if getattr(fn, "arguments", None):
                    entry["arguments"] += fn.arguments

    def finalized_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for idx in sorted(self._by_index):
            entry = self._by_index[idx]
            if not entry.get("name"):
                continue
            calls.append(
                {
                    "id": entry.get("id") or f"call_{idx}",
                    "name": entry["name"],
                    "arguments": entry.get("arguments") or "{}",
                }
            )
        return calls


async def generate_with_tool_loop(
    *,
    client: Any,
    model: str,
    temperature: float,
    extra_body: dict[str, Any],
    messages: list[dict[str, Any]],
    settings: ToolCallingSettings,
) -> AsyncIterator[LLMResponseChunk]:
    """Run a bounded OpenAI-compatible tool loop, yielding only assistant tokens."""

    executor = ToolExecutor(settings)
    tool_defs = executor.openai_tool_definitions()
    if not settings.tools_effectively_enabled() or not tool_defs:
        raise RuntimeError("tool_loop_invoked_without_tools")

    working = list(messages)
    deadline = time.monotonic() + settings.total_timeout_s
    accumulated_out = ""

    for round_idx in range(settings.max_rounds):
        if time.monotonic() > deadline:
            yield _final_chunk(
                accumulated_out + "\n（工具调用已达总时限，回答基于已有信息。）",
                accumulated_out,
            )
            return

        round_deadline = min(deadline, time.monotonic() + settings.round_timeout_s)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": working,
            "temperature": temperature,
            "stream": True,
            "tools": tool_defs,
            "tool_choice": "auto",
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        tool_acc = _ToolCallStreamAccumulator()
        round_text = ""
        finish_reason = None
        assistant_message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}

        try:
            stream = await asyncio.wait_for(
                client.chat.completions.create(**create_kwargs),
                timeout=max(1.0, round_deadline - time.monotonic()),
            )
        except TimeoutError:
            yield _final_chunk(
                accumulated_out + "\n（本轮工具请求超时。）",
                accumulated_out,
            )
            return

        async for chunk in stream:
            if time.monotonic() > round_deadline:
                yield _final_chunk(
                    accumulated_out + "\n（本轮工具请求超时。）",
                    accumulated_out,
                )
                return
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                token = delta.content
                round_text += token
                accumulated_out += token
                is_sentence_end = any(token.endswith(p) for p in SENTENCE_ENDERS)
                yield LLMResponseChunk(
                    token=token,
                    accumulated_text=accumulated_out,
                    is_sentence_end=is_sentence_end,
                    is_final=False,
                )
            if delta.tool_calls:
                tool_acc.ingest_delta(delta.tool_calls)

        calls = tool_acc.finalized_calls()
        if finish_reason == "tool_calls" or calls:
            assistant_message["content"] = round_text or None
            assistant_message["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c.get("arguments", "{}")},
                }
                for c in calls
            ]
            working.append(assistant_message)
            if not calls:
                yield _final_chunk(accumulated_out, accumulated_out)
                return
            tool_msgs = await executor.execute_many(calls)
            working.extend(tool_msgs)
            logger.info("tool_loop round=%d executed %d tool call(s)", round_idx + 1, len(calls))
            continue

        yield _final_chunk(accumulated_out or round_text, accumulated_out or round_text)
        return

    yield _final_chunk(
        (accumulated_out + "\n（已达到工具调用轮次上限。）").strip(),
        accumulated_out,
    )


def _final_chunk(text: str, accumulated: str) -> LLMResponseChunk:
    return LLMResponseChunk(
        token="",
        accumulated_text=text or accumulated,
        is_sentence_end=True,
        is_final=True,
    )


def append_capability_system_message(messages: list[dict[str, Any]], settings: ToolCallingSettings) -> list[dict[str, Any]]:
    note = settings.capability_system_note()
    out = list(messages)
    if out and out[0].get("role") == "system":
        prior = str(out[0].get("content", "") or "")
        out[0] = {**out[0], "content": f"{note}\n\n{prior}".strip()}
    else:
        out.insert(0, {"role": "system", "content": note})
    return out