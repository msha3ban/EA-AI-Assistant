from __future__ import annotations

import glob
import gzip
import hashlib
import logging
import math
import os
import subprocess
import wave
from pathlib import Path
from typing import Any

from .config import STTConfig
from .models import Segment

log = logging.getLogger(__name__)
SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 30
MAX_CHUNK_SAMPLES = MAX_CHUNK_SECONDS * SAMPLE_RATE
MAX_MERGE_GAP_SECONDS = 1.0
MAX_MERGE_GAP_SAMPLES = int(MAX_MERGE_GAP_SECONDS * SAMPLE_RATE)


class FfmpegAudio:
    def probe(self, path: str) -> float:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(out.stdout.strip())

    def normalize(
        self,
        path: str,
        output: str,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        command = ["ffmpeg", "-y", "-i", path]
        if start is not None:
            command.extend(["-ss", start])
        if end is not None:
            command.extend(["-to", end])
        command.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output])
        subprocess.run(command, check=True, capture_output=True)


def _preload_cuda_libraries() -> None:
    """Load cuBLAS/cuDNN from the NVIDIA pip wheels so ctranslate2 finds them.

    ctranslate2 4.8 needs CUDA 12 cuBLAS and cuDNN 9; the `stt` extra installs
    them as wheels, which the dynamic loader doesn't search by default.
    """
    import ctypes
    import importlib.util

    pending: list[str] = []
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            continue
        for location in (spec.submodule_search_locations or []) if spec else []:
            pending += sorted(glob.glob(os.path.join(location, "lib", "lib*.so*")))
    # Libraries depend on each other; retry until no further progress.
    while pending:
        failed = []
        for library in pending:
            try:
                ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(library)
        if len(failed) == len(pending):
            log.warning("Could not preload %d CUDA libraries", len(failed))
            return
        pending = failed


def _read_normalized_wav(path: str) -> Any:
    """Decode a 16 kHz mono 16-bit WAV to float32 samples in [-1, 1).

    faster-whisper 1.2.1 decodes file paths through a PyAV API that PyAV 19
    removed, so the adapter hands it samples from the ffmpeg-normalised WAV.
    """
    import numpy as np

    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError(
                "Speech-to-text input must be a 16 kHz mono 16-bit WAV; "
                f"got {w.getframerate()} Hz, {w.getnchannels()} channel(s), "
                f"{w.getsampwidth() * 8}-bit"
            )
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class FasterWhisper:
    def __init__(self) -> None:
        self.model_identifier = "unknown"
        self.detected_language: str | None = None
        self.language_probability: float | None = None

    def validate_compute_type(self, config: STTConfig) -> None:
        if config.device == "cuda":
            import ctranslate2

            supported = ctranslate2.get_supported_compute_types("cuda")
            if config.compute_type not in supported:
                raise ValueError(
                    f"Compute type {config.compute_type!r} is not supported on configured CUDA device; supported types: {', '.join(sorted(supported))}"
                )

    def transcribe(
        self, path: str, config: STTConfig
    ) -> tuple[list[Segment], str, str]:
        audio = _read_normalized_wav(path)
        _preload_cuda_libraries()
        from faster_whisper import WhisperModel, download_model

        local_path = Path(config.model).expanduser()
        model_path = (
            str(local_path.resolve())
            if local_path.is_dir()
            else download_model(
                config.model, local_files_only=True, cache_dir=config.download_root
            )
        )
        self.model_identifier = self._identifier(Path(model_path))
        if self.model_identifier == "unknown":
            log.warning(
                "Could not determine local speech-to-text model digest for %s",
                config.model,
            )
        model = WhisperModel(
            model_path,
            device=config.device,
            compute_type=config.compute_type,
            download_root=config.download_root,
            local_files_only=True,
        )
        try:
            kwargs: dict[str, Any] = {}
            if config.initial_prompt is not None:
                kwargs["initial_prompt"] = config.initial_prompt
            if config.hotwords is not None:
                kwargs["hotwords"] = config.hotwords
            raw, info = model.transcribe(
                audio,
                task="transcribe",
                language=None if config.language == "auto" else config.language,
                beam_size=config.beam_size,
                vad_filter=config.vad_filter,
                vad_parameters=config.vad_parameters,
                word_timestamps=True,
                **kwargs,
            )
            self.detected_language = getattr(info, "language", None)
            probability = getattr(info, "language_probability", None)
            self.language_probability = (
                float(probability) if probability is not None else None
            )
            segments = [
                Segment(
                    f"s{i:04d}",
                    float(s.start),
                    float(s.end),
                    s.text,
                    float(s.avg_logprob),
                    float(s.no_speech_prob),
                    float(s.compression_ratio),
                    word_probabilities=tuple(
                        round(float(word.probability), 4)
                        for word in (s.words or ())
                        if word.probability is not None
                    ),
                )
                for i, s in enumerate(raw, 1)
            ]
            compute_type = (
                getattr(getattr(model, "model", None), "compute_type", None)
                or config.compute_type
            )
            return segments, config.model, str(compute_type)
        finally:
            del model
            import gc

            gc.collect()

    def _identifier(self, model_path: Path) -> str:
        resolved = model_path.resolve()
        parts = resolved.parts
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts):
                return parts[index + 1]
        weights = resolved / "model.bin"
        if weights.is_file():
            digest = hashlib.sha256()
            with weights.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            return "sha256:" + digest.hexdigest()
        return "unknown"

    def release(self) -> None:
        import gc

        gc.collect()


class TranscribeCppSpeechToText:
    """Map token log probability, gzip ratio, and absent silence to Segment signals."""

    def __init__(self) -> None:
        self.model_identifier = "unknown"
        self.detected_language: str | None = None
        self.language_probability: float | None = None
        self.vocabulary_applied = False
        self._model: Any = None
        self._session: Any = None
        self._binding: Any = None

    def validate_compute_type(self, config: STTConfig) -> None:
        return None

    def transcribe(
        self, path: str, config: STTConfig
    ) -> tuple[list[Segment], str, str]:
        audio = _read_normalized_wav(path)
        self.release()
        try:
            self._load_model(config)
            vocabulary = self._choose_vocabulary(config)
            segments: list[Segment] = []
            for start, end in self._speech_chunks(audio, config):
                segment = self._transcribe_chunk(
                    audio, start, end, config, vocabulary, len(segments) + 1
                )
                if segment is not None:
                    segments.append(segment)
            self.detected_language = config.language
            return segments, config.model, config.backend
        except BaseException:
            self.release()
            raise

    def _load_model(self, config: STTConfig) -> None:
        try:
            import transcribe_cpp
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "transcribe_cpp binding unavailable; set PYTHONPATH to its "
                "bindings/python/src and TRANSCRIBE_LIBRARY to libtranscribe.so"
            ) from exc
        model_path = Path(config.model).expanduser().resolve()
        self.model_identifier = self._identifier(model_path)
        self._binding = transcribe_cpp
        self._model = transcribe_cpp.Model(str(model_path), backend=config.backend)
        self._session = self._model.session()

    def _choose_vocabulary(self, config: STTConfig) -> list[str] | None:
        vocabulary = list(config.vocabulary) or None
        supported = bool(self._model.supports("vocabulary"))
        self.vocabulary_applied = bool(vocabulary and supported)
        if vocabulary and not supported:
            log.warning(
                "Configured Vocabulary ignored: local speech-to-text model "
                "does not support vocabulary"
            )
        return vocabulary if self.vocabulary_applied else None

    def _speech_chunks(
        self, audio: Any, config: STTConfig
    ) -> list[tuple[int, int]]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        regions = (
            get_speech_timestamps(audio, VadOptions(**config.vad_parameters))
            if config.vad_filter
            else [{"start": 0, "end": len(audio)}]
        )
        return self._chunks(regions, len(audio))

    def _transcribe_chunk(
        self,
        audio: Any,
        start: int,
        end: int,
        config: STTConfig,
        vocabulary: list[str] | None,
        number: int,
    ) -> Segment | None:
        if end <= start:
            return None
        flags: list[str] = []
        try:
            result = self._session.run(
                audio[start:end],
                language=config.language,
                vocabulary=vocabulary,
                pnc="default",
            )
        except (
            self._binding.OutputTruncated,
            self._binding.OutputRepetition,
        ) as exc:
            result = exc.partial_result
            flags.append("repetition loop")
        value = result.text.strip()
        if not value:
            return None
        probabilities = [
            float(token.p) for token in result.tokens if token.p is not None
        ]
        avg_logprob = (
            sum(math.log(max(probability, 1e-10)) for probability in probabilities)
            / len(probabilities)
            if probabilities
            else 0.0
        )
        encoded = value.encode("utf-8")
        ratio = len(encoded) / len(gzip.compress(encoded)) if encoded else 0.0
        return Segment(
            f"s{number:04d}",
            start / SAMPLE_RATE,
            end / SAMPLE_RATE,
            value,
            avg_logprob,
            0.0,
            ratio,
            flags,
            confidence_signals=bool(probabilities),
        )

    @staticmethod
    def _chunks(
        regions: list[dict[str, int]], sample_count: int
    ) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        for region in regions:
            start = max(0, min(int(region["start"]), sample_count))
            end = max(start, min(int(region["end"]), sample_count))
            if chunks:
                start = max(start, chunks[-1][1])
            while start < end:
                if (
                    chunks
                    and start - chunks[-1][1] <= MAX_MERGE_GAP_SAMPLES
                    and end - chunks[-1][0] <= MAX_CHUNK_SAMPLES
                ):
                    chunks[-1] = (chunks[-1][0], end)
                    break
                stop = min(start + MAX_CHUNK_SAMPLES, end)
                chunks.append((start, stop))
                start = stop
        return chunks

    @staticmethod
    def _identifier(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as model:
            for block in iter(lambda: model.read(1024 * 1024), b""):
                digest.update(block)
        return "sha256:" + digest.hexdigest()

    def release(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._model is not None:
            self._model.close()
            self._model = None


class MachineSensors:
    def gpu_temperature(self) -> float | None:
        try:
            value = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0]
            return float(value)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def on_ac(self) -> bool | None:
        try:
            mains = []
            for path in glob.glob("/sys/class/power_supply/*"):
                with open(os.path.join(path, "type")) as handle:
                    if handle.read().strip() == "Mains":
                        mains.append(path)
            if not mains:
                return None
            values = []
            for path in mains:
                with open(os.path.join(path, "online")) as handle:
                    values.append(handle.read().strip() == "1")
            return any(values)
        except OSError:
            return None
