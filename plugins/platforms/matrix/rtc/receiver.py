"""LiveKit room lifecycle for MatrixRTC: join, subscribe, hear the user.

Inbound half of the duplex only. Publishing TTS, mapping utterances onto a gateway
session, and writing ``m.rtc.member`` so the bot shows up in a client's call UI are
each their own concern and are not wired here.

Two traps this module exists to encapsulate, both established against a live SFU:

* **Ask the SDK for 16 kHz mono.** The wire carries 48 kHz; ``AudioStream`` takes a
  ``sample_rate`` that reaches the Rust FFI, so the native resampler hands us exactly
  what Whisper wants. Resampling in Python — or shelling out to ffmpeg as the Discord
  receiver must — buys nothing here.
* **Sleep 0.5 s after ``disconnect()``.** The FFI's tokio worker is still draining when
  ``disconnect()`` returns; letting the event loop close over it aborts the process with
  a non-unwinding panic *after a completely successful run*, so the exit code lies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from .segmenter import CHANNELS, SAMPLE_RATE, TurnSegmenter, transcribe_pcm

logger = logging.getLogger(__name__)

LAZY_FEATURE = "platform.matrix_rtc"

# How often we ask the segmenter whether anyone has stopped talking. Matches the
# Discord voice loop; well under SILENCE_THRESHOLD, so the poll never sets the latency.
POLL_INTERVAL = 0.2

# The FFI worker outlives disconnect() by a hair. See the module docstring.
FFI_DRAIN_DELAY = 0.5


def livekit_available() -> bool:
    """True when the LiveKit SDK can be imported (or its lazy feature is satisfied)."""
    try:
        from tools.lazy_deps import is_available
        return is_available(LAZY_FEATURE)
    except Exception:
        try:
            import livekit.rtc  # noqa: F401
            return True
        except ImportError:
            return False


class MatrixRTCReceiver:
    """Joins a LiveKit room and calls *on_transcript* once per completed utterance.

    ``on_transcript(identity, transcript)`` is awaited on the receiver's own loop;
    ``identity`` is the LiveKit participant identity, which the Matrix JWT service
    derives as ``{matrix_user_id}:{device_id}``.

    *is_authorized(identity)* is consulted once per utterance *before* transcription, so
    audio from a participant the operator never allowed is never sent to Whisper at all.
    Omitting it transcribes every speaker and leaves the allowlist entirely to the caller.
    """

    def __init__(self, on_transcript: Callable[[str, str], Awaitable[None]],
                 segmenter: Optional[TurnSegmenter] = None,
                 sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS,
                 is_authorized: Optional[Callable[[str], bool]] = None):
        self._on_transcript = on_transcript
        self._is_authorized = is_authorized
        self.sample_rate = sample_rate
        self.channels = channels
        self.segmenter = segmenter or TurnSegmenter(sample_rate, channels)
        self._room: Any = None
        self._tasks: set[asyncio.Task] = set()
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False

    # --- lifecycle ---

    async def connect(self, sfu_url: str, jwt: str) -> None:
        """Join the SFU and start listening. *jwt* comes from ``focus.fetch_livekit_credentials``."""
        from tools.lazy_deps import ensure
        await asyncio.to_thread(ensure, LAZY_FEATURE, prompt=False)
        from livekit import rtc

        room = rtc.Room()

        @room.on("track_subscribed")
        def _on_track(track, publication, participant):  # noqa: ARG001 - SDK signature
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            logger.info("MatrixRTC: subscribed to audio from %s", participant.identity)
            self._spawn(self._drain_track(rtc, track, participant.identity))

        await room.connect(sfu_url, jwt, options=rtc.RoomOptions(auto_subscribe=True))
        self._room = room
        self._running = True
        logger.info("MatrixRTC: joined as %s", room.local_participant.identity)
        self._poll_task = asyncio.create_task(self._poll_silence())

    async def close(self) -> None:
        """Leave the room, emit whatever was still buffered, and let the FFI drain."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task, return_exceptions=True)
            self._poll_task = None
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        await self._emit(self.segmenter.flush_pending())
        if self._room is not None:
            await self._room.disconnect()
            self._room = None
            # Not cosmetic: without this the process aborts on a successful run.
            await asyncio.sleep(FFI_DRAIN_DELAY)

    # --- internals ---

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drain_track(self, rtc, track, identity: str) -> None:
        """Feed one remote track's PCM into the segmenter until it ends."""
        stream = rtc.AudioStream(
            track, sample_rate=self.sample_rate, num_channels=self.channels)
        try:
            async for event in stream:
                self.segmenter.feed(identity, bytes(event.frame.data))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("MatrixRTC: audio stream for %s ended: %s", identity, exc)
        finally:
            await stream.aclose()

    async def _poll_silence(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(POLL_INTERVAL)
                await self._emit(self.segmenter.check_silence())
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("MatrixRTC: silence poll loop failed", exc_info=True)

    async def _emit(self, utterances) -> None:
        for identity, pcm in utterances:
            if self._is_authorized is not None and not self._is_authorized(identity):
                logger.info("MatrixRTC: discarding audio from %s before transcription", identity)
                continue
            try:
                transcript = await asyncio.to_thread(
                    transcribe_pcm, pcm, self.sample_rate, self.channels)
            except Exception as exc:
                logger.warning("MatrixRTC: transcription failed for %s: %s", identity, exc)
                continue
            if not transcript:
                continue
            logger.info("MatrixRTC voice input from %s: %s", identity, transcript[:100])
            try:
                await self._on_transcript(identity, transcript)
            except Exception:
                logger.error("MatrixRTC: transcript callback failed", exc_info=True)
