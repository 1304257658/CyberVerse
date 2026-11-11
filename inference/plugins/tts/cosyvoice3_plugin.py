"""Local Fun-CosyVoice3-0.5B TTS plugin, served by a local worker process.

Same contract as the cloud TTS plugins: consume a text stream, yield 16 kHz
float32 ``AudioChunk``s rechunked with ``AudioRechunker`` for the avatar.

Streaming: incoming text is cut into sentences (。！？!?；;… / newline, long
sentences additionally at commas); each sentence is synthesized with
CosyVoice ``stream=True`` and audio packets are forwarded as soon as they are
produced.  The zero-shot reference voice is cached once when the worker
starts.  Synthesis runs in ``local_workers/cosyvoice3_worker.py`` inside the
CosyVoice venv (see ``inference/plugins/local_worker_client.py`` for why).
"""

import base64
import logging
import os
import re
import time
from typing import AsyncIterator

import numpy as np

from inference.core.types import AudioChunk, PluginConfig, TTSRequestConfig
from inference.plugins.local_worker_client import LocalModelWorker
from inference.plugins.tts.base import AudioRechunker, TTSPlugin

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;…\n])|(?<=[.])(?=\s)")
_SOFT_BREAK = re.compile(r"(?<=[，,、：:])")


def split_sentences(text: str, max_chars: int = 80) -> tuple[list[str], str]:
    """Return (complete sentences, unfinished tail)."""
    parts = _SENTENCE_END.split(text)
    tail = parts.pop() if parts else ""
    out: list[str] = []
    for part in parts:
        out.extend(_split_long(part, max_chars))
    return [s for s in (x.strip() for x in out) if s], tail


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(sentence) <= max_chars:
        return [sentence]
    out, cur = [], ""
    for piece in _SOFT_BREAK.split(sentence):
        if cur and len(cur) + len(piece) > max_chars:
            out.append(cur)
            cur = ""
        cur += piece
        while len(cur) > max_chars:
            out.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        out.append(cur)
    return out


class CosyVoice3TTSPlugin(TTSPlugin):
    name = "tts.cosyvoice3"

    def __init__(self) -> None:
        self.worker: LocalModelWorker | None = None
        self.target_sample_rate = 16000
        self.rechunk_samples = 17920
        self.speed = 1.0
        self.max_segment_chars = 80
        self.min_segment_chars = 2

    async def initialize(self, config: PluginConfig) -> None:
        p = config.params
        self.target_sample_rate = int(p.get("target_sample_rate", self.target_sample_rate))
        self.rechunk_samples = int(p.get("rechunk_samples", self.rechunk_samples))
        self.speed = float(p.get("speed", self.speed))
        self.max_segment_chars = int(p.get("max_segment_chars", self.max_segment_chars))
        self.min_segment_chars = int(p.get("min_segment_chars", self.min_segment_chars))

        python = str(p.get("python", ""))
        repo_dir = os.path.abspath(str(p.get("repo_dir", "")))
        model_dir = os.path.abspath(str(p.get("model_dir", "")))
        prompt_wav = os.path.abspath(str(p.get("prompt_wav", "")))
        if python:
            for label, path in (("repo_dir", repo_dir), ("model_dir", model_dir)):
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"CosyVoice3 {label} not found: {path!r}")
            if not os.path.isfile(prompt_wav):
                raise FileNotFoundError(f"CosyVoice3 prompt_wav not found: {prompt_wav!r}")
        self.worker = LocalModelWorker(
            name="cosyvoice3",
            script="cosyvoice3_worker.py",
            args=[
                "--repo", repo_dir,
                "--model-dir", model_dir,
                "--prompt-wav", prompt_wav,
                "--prompt-text", str(p.get("prompt_text", "")),
                "--spk-id", str(p.get("voice", "cyberverse_default") or "cyberverse_default"),
                "--target-sample-rate", str(self.target_sample_rate),
                "--fp16", "1" if p.get("fp16", False) else "0",
            ],
            socket_path=str(p.get("socket_path", "/tmp/cyberverse-cosyvoice3.sock")),
            python=python,
            startup_timeout_s=float(p.get("startup_timeout_s", 900)),
            env={k: str(v) for k, v in (p.get("env") or {}).items()},
            cwd=repo_dir or None,
        )
        await self.worker.start()

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        request_config: TTSRequestConfig | None = None,
    ) -> AsyncIterator[AudioChunk]:
        if self.worker is None:
            raise RuntimeError("CosyVoice3 TTS plugin is not initialized")
        rechunker = (
            AudioRechunker(chunk_samples=self.rechunk_samples, sample_rate=self.target_sample_rate)
            if self.rechunk_samples > 0
            else None
        )
        t0 = time.perf_counter()
        first_logged = False
        pending = ""
        carry = ""

        async def synth(sentence: str) -> AsyncIterator[AudioChunk]:
            nonlocal first_logged
            async for msg in self.worker.stream({"op": "tts", "text": sentence, "speed": self.speed}):
                if msg.get("done"):
                    logger.info(
                        "CosyVoice3 segment chars=%d first_packet=%.0fms total=%.0fms audio=%.0fms rtf=%s",
                        len(sentence), msg.get("first_packet_ms", 0), msg.get("total_ms", 0),
                        msg.get("audio_ms", 0), msg.get("rtf"),
                    )
                    return
                audio = np.frombuffer(base64.b64decode(msg["audio"]), dtype=np.float32)
                if rechunker is not None:
                    chunks = rechunker.feed(audio)
                else:
                    chunks = [AudioChunk(
                        data=audio.tobytes(),
                        sample_rate=int(msg.get("sample_rate", self.target_sample_rate)),
                        duration_ms=int(audio.size * 1000 / self.target_sample_rate),
                    )]
                for chunk in chunks:
                    if not first_logged:
                        first_logged = True
                        logger.info("CosyVoice3 first audio chunk after %.0fms", (time.perf_counter() - t0) * 1000)
                    yield chunk

        async for text in text_stream:
            if not text:
                continue
            pending += text
            sentences, pending = split_sentences(pending, self.max_segment_chars)
            for sentence in sentences:
                sentence = carry + sentence
                carry = ""
                if len(sentence) < self.min_segment_chars:
                    carry = sentence
                    continue
                async for chunk in synth(sentence):
                    yield chunk

        tail = (carry + pending).strip()
        if tail:
            for sentence in _split_long(tail, self.max_segment_chars):
                if sentence.strip():
                    async for chunk in synth(sentence.strip()):
                        yield chunk

        if rechunker is not None:
            final_chunk = rechunker.flush()
            if final_chunk:
                yield final_chunk

    async def shutdown(self) -> None:
        if self.worker is not None:
            await self.worker.stop()
            self.worker = None
        logger.info("CosyVoice3 TTS plugin shut down")