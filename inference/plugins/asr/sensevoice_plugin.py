"""Local SenseVoiceSmall ASR plugin (FunASR), served by a local worker process.

Same contract as ``WhisperASRPlugin``: consume a stream of mic audio, detect
utterance boundaries with an energy threshold, and yield a final
``TranscriptEvent`` per utterance.  Recognition runs in
``local_workers/sensevoice_worker.py`` inside its own venv (see
``inference/plugins/local_worker_client.py`` for why).

Input audio from the Go server is 16 kHz mono signed 16-bit PCM
(``input_format: pcm_s16le``); ``float32`` is accepted too.
"""

import base64
import collections
import logging
import os
from typing import AsyncIterator

import numpy as np

from inference.core.types import ASRRequestConfig, PluginConfig, TranscriptEvent
from inference.plugins.asr.base import ASRPlugin
from inference.plugins.local_worker_client import LocalModelWorker

logger = logging.getLogger(__name__)


class SenseVoiceASRPlugin(ASRPlugin):
    name = "asr.sensevoice"

    def __init__(self) -> None:
        self.worker: LocalModelWorker | None = None
        self.language = "auto"
        self.input_format = "pcm_s16le"
        self.sample_rate = 16000
        self._silence_threshold = 0.01
        self._silence_duration = 0.6
        self._min_speech_seconds = 0.3
        self._max_utterance_seconds = 20.0
        self._pre_roll_seconds = 0.3

    async def initialize(self, config: PluginConfig) -> None:
        p = config.params
        self.language = str(p.get("language", self.language) or "auto")
        self.input_format = str(p.get("input_format", self.input_format)).lower()
        self.sample_rate = int(p.get("sample_rate", self.sample_rate))
        self._silence_threshold = float(p.get("silence_threshold", self._silence_threshold))
        self._silence_duration = float(p.get("silence_duration", self._silence_duration))
        self._min_speech_seconds = float(p.get("min_speech_seconds", self._min_speech_seconds))
        self._max_utterance_seconds = float(p.get("max_utterance_seconds", self._max_utterance_seconds))
        self._pre_roll_seconds = float(p.get("pre_roll_seconds", self._pre_roll_seconds))

        model_dir = os.path.abspath(str(p.get("model_dir", "")))
        python = str(p.get("python", ""))
        if python and not os.path.isdir(model_dir):
            raise FileNotFoundError(f"SenseVoice model_dir not found: {model_dir!r}")
        self.worker = LocalModelWorker(
            name="sensevoice",
            script="sensevoice_worker.py",
            args=[
                "--model-dir", model_dir,
                "--device", str(p.get("device", "cuda:0")),
                "--use-itn", "1" if p.get("use_itn", True) else "0",
            ],
            socket_path=str(p.get("socket_path", "/tmp/cyberverse-sensevoice.sock")),
            python=python,
            startup_timeout_s=float(p.get("startup_timeout_s", 600)),
            env={k: str(v) for k, v in (p.get("env") or {}).items()},
        )
        await self.worker.start()

    def _to_float32(self, chunk: bytes) -> np.ndarray:
        if self.input_format in ("float32", "f32", "pcm_f32le"):
            return np.frombuffer(chunk, dtype=np.float32)
        usable = len(chunk) - (len(chunk) % 2)
        return np.frombuffer(chunk[:usable], dtype=np.int16).astype(np.float32) / 32768.0

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        request_config: ASRRequestConfig | None = None,
    ) -> AsyncIterator[TranscriptEvent]:
        language = (request_config.language if request_config else "") or self.language
        sr = self.sample_rate
        silence_limit = int(self._silence_duration * sr)
        min_speech = int(self._min_speech_seconds * sr)
        max_samples = int(self._max_utterance_seconds * sr)
        pre_roll_limit = int(self._pre_roll_seconds * sr)

        pre_roll: collections.deque[np.ndarray] = collections.deque()
        pre_roll_samples = 0
        segments: list[np.ndarray] = []
        seg_samples = 0
        voiced = 0
        silence = 0
        in_speech = False

        async for chunk_bytes in audio_stream:
            audio = self._to_float32(chunk_bytes)
            if audio.size == 0:
                continue
            rms = float(np.sqrt(np.mean(audio**2)))
            loud = rms >= self._silence_threshold

            if not in_speech:
                pre_roll.append(audio)
                pre_roll_samples += audio.size
                while pre_roll and pre_roll_samples - pre_roll[0].size >= pre_roll_limit:
                    pre_roll_samples -= pre_roll.popleft().size
                if loud:
                    in_speech = True
                    segments = list(pre_roll)
                    seg_samples = pre_roll_samples
                    voiced = audio.size
                    silence = 0
                    pre_roll.clear()
                    pre_roll_samples = 0
                continue

            segments.append(audio)
            seg_samples += audio.size
            if loud:
                voiced += audio.size
                silence = 0
            else:
                silence += audio.size

            if silence >= silence_limit or seg_samples >= max_samples:
                if voiced >= min_speech:
                    event = await self._transcribe(np.concatenate(segments), language)
                    if event is not None:
                        yield event
                segments, seg_samples, voiced, silence, in_speech = [], 0, 0, 0, False

        if in_speech and voiced >= min_speech:
            event = await self._transcribe(np.concatenate(segments), language)
            if event is not None:
                yield event

    async def _transcribe(self, audio: np.ndarray, language: str) -> TranscriptEvent | None:
        if self.worker is None:
            return None
        result = await self.worker.call(
            {
                "op": "asr",
                "audio": base64.b64encode(audio.astype(np.float32).tobytes()).decode(),
                "format": "float32",
                "sample_rate": self.sample_rate,
                "language": language,
            }
        )
        text = str(result.get("text", "")).strip()
        logger.info(
            "SenseVoice utterance audio=%.0fms asr=%.0fms lang=%s text=%r",
            result.get("audio_ms", 0.0), result.get("elapsed_ms", 0.0), result.get("language", ""), text,
        )
        if not text:
            return None
        lang = str(result.get("language", "")).strip()
        return TranscriptEvent(
            text=text,
            is_final=True,
            language="" if lang in ("nospeech", "auto") else lang,
            confidence=1.0,
        )

    async def shutdown(self) -> None:
        if self.worker is not None:
            await self.worker.stop()
            self.worker = None
        logger.info("SenseVoice ASR plugin shut down")