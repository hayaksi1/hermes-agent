"""Turn segmentation and transcription for MatrixRTC audio.

Deliberately knows nothing about LiveKit or Matrix: it takes raw PCM keyed by a
speaker label and hands back completed utterances, so it is testable with synthetic
audio and no network.

The timers are the ones Discord voice channels have been running in production
(``plugins/platforms/discord/adapter.py`` ``VoiceReceiver``): 1.5 s of silence ends an
utterance, and anything under 0.5 s is noise. Only that logic is shared — none of
Discord's RTP/SSRC/DAVE machinery applies here, because LiveKit delivers decoded PCM
already attributed to a participant identity.

Audio arrives as 16 kHz mono s16 because ``receiver.py`` asks the LiveKit SDK for that
rate (the wire is 48 kHz; the SDK's native resampler does the conversion). That is also
what Whisper wants, so ``pcm_to_wav`` is a stdlib ``wave`` write with no ffmpeg in the
path — unlike the Discord receiver, which must shell out to convert 48 kHz stereo.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Ported verbatim from Discord's VoiceReceiver — same speech, same ears.
SILENCE_THRESHOLD = 1.5     # seconds of silence -> end of utterance
MIN_SPEECH_DURATION = 0.5   # minimum seconds to process (skip noise)

# What we ask LiveKit to deliver, and therefore what the buffers hold.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # s16


def _rtc_config() -> dict:
    """``matrix.rtc`` from config.yaml. Behavioural knobs live there, never in ``.env``."""
    try:
        from hermes_cli.config import read_raw_config_readonly
        matrix_cfg = (read_raw_config_readonly() or {}).get("matrix") or {}
        return matrix_cfg.get("rtc") or {}
    except Exception as exc:  # config unreadable -> ship the defaults, don't crash the call
        logger.debug("MatrixRTC: config read failed, using defaults: %s", exc)
        return {}


def _positive_float(raw, fallback: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


class TurnSegmenter:
    """Buffers PCM per speaker and releases an utterance once they stop talking.

    Thread-safe: ``feed`` runs on whatever task drains the LiveKit audio stream while
    ``check_silence`` runs on the polling loop, exactly as on Discord.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS,
                 silence_threshold: Optional[float] = None,
                 min_speech_duration: Optional[float] = None):
        cfg = _rtc_config()
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = _positive_float(
            silence_threshold if silence_threshold is not None
            else cfg.get("silence_threshold"), SILENCE_THRESHOLD)
        self.min_speech_duration = _positive_float(
            min_speech_duration if min_speech_duration is not None
            else cfg.get("min_speech_duration"), MIN_SPEECH_DURATION)
        self._lock = threading.Lock()
        self._buffers: dict[str, bytearray] = defaultdict(bytearray)
        self._last_frame_time: dict[str, float] = {}

    # --- ingest ---

    def feed(self, identity: str, pcm: bytes, now: Optional[float] = None) -> None:
        """Append decoded PCM for *identity*. *now* is injectable so tests need no clock."""
        if not pcm:
            return
        stamp = time.monotonic() if now is None else now
        with self._lock:
            self._buffers[identity].extend(pcm)
            self._last_frame_time[identity] = stamp

    def _duration(self, buf) -> float:
        return len(buf) / (self.sample_rate * self.channels * SAMPLE_WIDTH)

    # --- release ---

    def check_silence(self, now: Optional[float] = None) -> list[tuple[str, bytes]]:
        """Return ``(identity, pcm)`` for every speaker who has gone quiet long enough."""
        stamp = time.monotonic() if now is None else now
        completed: list[tuple[str, bytes]] = []
        with self._lock:
            for identity in list(self._buffers):
                silence = stamp - self._last_frame_time.get(identity, stamp)
                buf = self._buffers[identity]
                if silence < self.silence_threshold:
                    continue
                if self._duration(buf) >= self.min_speech_duration:
                    completed.append((identity, bytes(buf)))
                    self._buffers[identity] = bytearray()
                    self._last_frame_time.pop(identity, None)
                elif silence >= self.silence_threshold * 2:
                    # Too short to be speech and long since abandoned: a cough, a door,
                    # a half-frame on join. Dropping it is what stops the map growing
                    # one dead entry per noise burst for the life of the call.
                    self._buffers.pop(identity, None)
                    self._last_frame_time.pop(identity, None)
        return completed

    def flush_pending(self) -> list[tuple[str, bytes]]:
        """Drain every buffer, returning the ones long enough to be speech. Used on leave."""
        completed: list[tuple[str, bytes]] = []
        with self._lock:
            for identity, buf in list(self._buffers.items()):
                if self._duration(buf) >= self.min_speech_duration:
                    completed.append((identity, bytes(buf)))
                self._buffers.pop(identity, None)
                self._last_frame_time.pop(identity, None)
        return completed

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._last_frame_time.clear()


def pcm_to_wav(pcm: bytes, output_path: str, sample_rate: int = SAMPLE_RATE,
               channels: int = CHANNELS) -> str:
    """Wrap raw s16 PCM in a WAV container. Stdlib only — no ffmpeg on this path."""
    import wave
    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output_path


def transcribe_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE,
                   channels: int = CHANNELS) -> Optional[str]:
    """PCM -> WAV -> Whisper -> text, or None when there is nothing worth passing on.

    Blocking (STT is CPU-bound); callers on the event loop must use ``asyncio.to_thread``.
    """
    import os
    import tempfile
    from tools.transcription_tools import transcribe_audio
    from tools.voice_mode_transcript import is_whisper_hallucination

    handle = tempfile.NamedTemporaryFile(suffix=".wav", prefix="matrix_rtc_", delete=False)
    wav_path = handle.name
    handle.close()
    try:
        pcm_to_wav(pcm, wav_path, sample_rate, channels)
        result = transcribe_audio(wav_path, source="voice_mode")
        if not result.get("success"):
            logger.debug("MatrixRTC transcription failed: %s", result.get("error"))
            return None
        transcript = (result.get("transcript") or "").strip()
        # Whisper invents "Thank you." / subtitle credits out of near-silence; the same
        # filter the CLI and Discord voice paths use keeps that out of the session.
        if not transcript or is_whisper_hallucination(transcript):
            return None
        return transcript
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
