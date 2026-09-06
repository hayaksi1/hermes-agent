"""Headless MatrixRTC (MSC4143) voice participation over LiveKit.

``focus`` runs the JWT exchange, ``receiver`` owns the LiveKit room, ``segmenter``
turns decoded PCM into transcripts. Import ``receiver`` only where the LiveKit SDK is
expected; ``focus`` and ``segmenter`` are import-safe without it.
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
