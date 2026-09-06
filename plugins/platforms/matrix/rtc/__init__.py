"""Headless MatrixRTC (MSC4143) voice participation over LiveKit.

``focus`` runs the JWT exchange, ``receiver`` owns the LiveKit room, ``segmenter``
turns decoded PCM into transcripts, ``session`` lands those transcripts on the room's
gateway session. Only ``focus`` and ``segmenter`` are re-exported here, because they
are the two that import nothing heavier than the stdlib: reach for ``receiver`` only
where the LiveKit SDK is expected, and ``session`` only where the gateway is loaded.
"""

from .focus import MatrixRTCError, fetch_livekit_credentials
from .segmenter import (
    MIN_SPEECH_DURATION, SILENCE_THRESHOLD, TurnSegmenter, pcm_to_wav, transcribe_pcm)

__all__ = [
    "MIN_SPEECH_DURATION",
    "SILENCE_THRESHOLD",
    "MatrixRTCError",
    "TurnSegmenter",
    "fetch_livekit_credentials",
    "pcm_to_wav",
    "transcribe_pcm",
]
