#!/usr/bin/env python3
"""
Operator - Pipecat voice agent for Asterisk via AudioSocket

Uses Pipecat for conversation management (VAD, interruptions, turn detection)
with a custom AudioSocket transport to bridge Asterisk phone audio.

Usage:
    source .venv/bin/activate
    export DEEPGRAM_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export DEEPINFRA_API_KEY=...  # optional, for Kokoro TTS
    python agent.py
"""

import asyncio
import io
import os
import struct
import sys
import time
import wave
from typing import AsyncGenerator, Optional

import httpx
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.deepgram import DeepgramSTTService
from deepgram import LiveOptions
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.tts_service import TTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIOSOCKET_HOST = "127.0.0.1"
AUDIOSOCKET_PORT = 9092

# AudioSocket protocol
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

ASTERISK_RATE = 8000

SYSTEM_PROMPT = """\
You are a telephone operator. You speak in a calm, professional, \
mid-century American style. You are helpful, efficient, and slightly \
formal without being cold.

Keep responses SHORT. You are speaking on a phone. Long responses waste \
the caller's time. One to two sentences is ideal. Three is the maximum.

If you don't understand, say "I'm sorry, could you repeat that?" \
Never guess at unclear requests.

The current date and time is {now}.

You can transfer the caller to the following extensions:
- Extension 1: Hello world greeting (French)
- Extension 2: Echo test (caller hears themselves back)
- Extension 4: Music on hold
- Extension 5: Congratulations message
- Extension 6: CISM 89.3 — Montreal college radio
- Extension 7: KEXP — Seattle radio
- Extension 8: The Gamut radio

When the caller asks to be connected to a service (radio, music, echo test, \
etc.), use the transfer_call tool. Confirm what you're connecting them to \
before transferring. After transferring, the call leaves your hands — say a \
brief goodbye before using the tool.\
"""

TOOLS = [
    {
        "name": "transfer_call",
        "description": "Transfer the caller to another extension on the phone system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "extension": {
                    "type": "string",
                    "description": "The extension number to transfer to (e.g. '7' for KEXP)",
                }
            },
            "required": ["extension"],
        },
    }
]


# ---------------------------------------------------------------------------
# Custom AudioSocket Transport
# ---------------------------------------------------------------------------

class AudioSocketInputTransport(BaseInputTransport):
    """Reads audio from an Asterisk AudioSocket TCP connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._reader = reader
        self._receive_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._receive_task = self.create_task(self._receive_loop())

    async def stop(self, frame: EndFrame):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await super().cancel(frame)

    async def _receive_loop(self):
        try:
            while True:
                header = await self._reader.readexactly(3)
                msg_type = header[0]
                msg_len = struct.unpack(">H", header[1:3])[0]
                payload = (
                    await self._reader.readexactly(msg_len) if msg_len > 0 else b""
                )

                if msg_type == MSG_HANGUP or msg_type == MSG_ERROR:
                    logger.info("AudioSocket: hangup/error received")
                    await self.push_frame(EndFrame())
                    return
                elif msg_type == MSG_AUDIO:
                    frame = InputAudioRawFrame(
                        audio=payload,
                        sample_rate=ASTERISK_RATE,
                        num_channels=1,
                    )
                    await self.push_audio_frame(frame)

        except asyncio.IncompleteReadError:
            logger.info("AudioSocket: connection closed")
            await self.push_frame(EndFrame())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"AudioSocket input error: {e}")
            await self.push_frame(EndFrame())


class AudioSocketOutputTransport(BaseOutputTransport):
    """Writes audio to an Asterisk AudioSocket TCP connection."""

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._writer = writer
        self._playback_start: float = 0
        self._samples_sent: int = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._playback_start = 0
        self._samples_sent = 0

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Send audio chunk via AudioSocket protocol with real-time pacing."""
        try:
            data = frame.audio
            rate = frame.sample_rate or ASTERISK_RATE

            # Reset clock on first chunk of a new utterance
            if self._playback_start == 0:
                self._playback_start = time.monotonic()
                self._samples_sent = 0

            header = struct.pack(">BH", MSG_AUDIO, len(data))
            self._writer.write(header + data)
            await self._writer.drain()

            # Pace to real-time
            self._samples_sent += len(data) // 2  # 16-bit = 2 bytes/sample
            target_time = self._samples_sent / rate
            elapsed = time.monotonic() - self._playback_start
            sleep = target_time - elapsed
            if sleep > 0:
                await asyncio.sleep(sleep)

            return True
        except Exception as e:
            logger.error(f"AudioSocket write error: {e}")
            return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Reset playback clock between utterances / on interruption
        if isinstance(frame, (InterruptionFrame, BotStoppedSpeakingFrame)):
            self._playback_start = 0
            self._samples_sent = 0
        await super().process_frame(frame, direction)


    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        try:
            self._writer.write(struct.pack(">BH", MSG_HANGUP, 0))
            await self._writer.drain()
        except Exception:
            pass

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)


class AudioSocketTransport(BaseTransport):
    """Pipecat transport bridging Asterisk AudioSocket."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        params: TransportParams,
    ):
        super().__init__()
        self._input = AudioSocketInputTransport(
            reader, params, name="AudioSocketInput"
        )
        self._output = AudioSocketOutputTransport(
            writer, params, name="AudioSocketOutput"
        )

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output


# ---------------------------------------------------------------------------
# Custom TTS: Kokoro via DeepInfra, espeak-ng fallback
# ---------------------------------------------------------------------------

class KokoroTTSService(TTSService):
    """Kokoro TTS via DeepInfra with espeak-ng fallback."""

    KOKORO_URL = "https://api.deepinfra.com/v1/openai/audio/speech"
    KOKORO_RATE = 24000

    def __init__(self, *, api_key: str = "", voice: str = "af_heart", **kwargs):
        super().__init__(sample_rate=self.KOKORO_RATE, **kwargs)
        self._api_key = api_key
        self._voice = voice

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # Try Kokoro (streaming)
        if self._api_key:
            try:
                async for chunk in self._kokoro_stream(text):
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.KOKORO_RATE,
                        num_channels=1,
                    )
                return
            except Exception as e:
                logger.warning(f"Kokoro TTS failed: {e}")

        # Fallback: espeak-ng
        audio, rate = await self._espeak(text)
        yield TTSAudioRawFrame(
            audio=audio,
            sample_rate=rate,
            num_channels=1,
        )

    async def _kokoro_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.KOKORO_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "hexgrad/Kokoro-82M",
                    "input": text,
                    "voice": self._voice,
                    "response_format": "pcm",
                },
                timeout=15.0,
            ) as resp:
                resp.raise_for_status()
                # 4800 bytes = 100ms at 24kHz 16-bit mono
                async for chunk in resp.aiter_bytes(chunk_size=4800):
                    if chunk:
                        yield chunk

    async def _espeak(self, text: str) -> tuple[bytes, int]:
        proc = await asyncio.create_subprocess_exec(
            "espeak-ng", "--stdout", "-v", "en-us", "-s", "150", text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {stderr.decode()}")
        with io.BytesIO(stdout) as buf:
            with wave.open(buf, "rb") as w:
                return w.readframes(w.getnframes()), w.getframerate()


# ---------------------------------------------------------------------------
# Call handler
# ---------------------------------------------------------------------------

class TransferWatcher(FrameProcessor):
    """Watches for BotStoppedSpeakingFrame and triggers pending transfers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pending_transfer: Optional[str] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame) and self.pending_transfer:
            ext = self.pending_transfer
            self.pending_transfer = None
            asyncio.create_task(_delayed_transfer(ext, delay=0.5))
        await self.push_frame(frame, direction)


async def _delayed_transfer(ext: str, delay: float = 5):
    """Wait for goodbye TTS to finish, then redirect the Asterisk channel."""
    await asyncio.sleep(delay)
    await do_transfer(ext)


async def handle_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):

    logger.info("New AudioSocket connection")

    # Read UUID message
    try:
        header = await reader.readexactly(3)
        msg_type = header[0]
        msg_len = struct.unpack(">H", header[1:3])[0]
        payload = await reader.readexactly(msg_len) if msg_len > 0 else b""
        if msg_type == MSG_UUID:
            call_uuid = payload.decode("utf-8", errors="replace").strip("\x00")
            logger.info(f"Call UUID: {call_uuid}")
    except Exception as e:
        logger.error(f"Failed to read UUID: {e}")
        writer.close()
        return

    # -- Transport --
    transport = AudioSocketTransport(
        reader,
        writer,
        TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=ASTERISK_RATE,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            audio_out_sample_rate=ASTERISK_RATE,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6)),
        ),
    )

    # -- STT --
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        sample_rate=ASTERISK_RATE,
        live_options=LiveOptions(
            model="nova-2",
            language="en",
            encoding="linear16",
            channels=1,
            sample_rate=ASTERISK_RATE,
            interim_results=True,
            smart_format=False,
            punctuate=True,
            profanity_filter=False,
        ),
    )

    # -- LLM --
    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
    )

    # -- TTS --
    tts = KokoroTTSService(
        api_key=os.environ.get("DEEPINFRA_API_KEY", ""),
    )

    # -- Context --
    now = time.strftime("%A, %B %d, %Y at %I:%M %p")
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT.format(now=now)}],
        tools=TOOLS,
    )
    context_aggregator = llm.create_context_aggregator(context)

    # -- Transfer watcher --
    transfer_watcher = TransferWatcher(name="TransferWatcher")

    # -- Tool: transfer_call --
    async def on_transfer_call(params: FunctionCallParams):
        ext = params.arguments.get("extension", "")
        valid = {"1", "2", "4", "5", "6", "7", "8"}
        if ext not in valid:
            await params.result_callback(
                f"Invalid extension {ext}. Valid: {', '.join(sorted(valid))}"
            )
            return
        await params.result_callback(f"Transferring to extension {ext}.")
        # TransferWatcher will do the redirect when bot finishes speaking.
        transfer_watcher.pending_transfer = ext

    llm.register_function("transfer_call", on_transfer_call)

    # -- Pipeline --
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            transfer_watcher,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False,
        ),
    )

    # Queue greeting
    await task.queue_frames([TTSSpeakFrame("Operator. How may I help you?")])

    # Run pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info("Pipeline finished")

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    logger.info("Call ended\n")


async def do_transfer(extension: str):
    """Redirect the Asterisk channel to another extension."""
    proc = await asyncio.create_subprocess_exec(
        "sudo", "asterisk", "-rx", "core show channels concise",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    channel_name = None
    for line in stdout.decode().splitlines():
        if "PJSIP/100-" in line and "AudioSocket" in line:
            channel_name = line.split("!")[0]
            break

    if not channel_name:
        logger.error("Transfer failed: could not find active channel")
        return

    logger.info(f"Redirecting {channel_name} → extension {extension}")
    proc = await asyncio.create_subprocess_exec(
        "sudo", "asterisk", "-rx",
        f"channel redirect {channel_name} internal,{extension},1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    missing = []
    for key in ["DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY"]:
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print(f"\n  Missing required env vars: {', '.join(missing)}")
        print(f"  Set them and try again.\n")
        sys.exit(1)

    has_kokoro = bool(os.environ.get("DEEPINFRA_API_KEY"))
    logger.info(f"TTS: {'Kokoro (DeepInfra)' if has_kokoro else 'espeak-ng (local)'}")

    server = await asyncio.start_server(handle_call, AUDIOSOCKET_HOST, AUDIOSOCKET_PORT)
    logger.info(f"Operator listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")
    logger.info("Dial 0 from the phone to connect.")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
