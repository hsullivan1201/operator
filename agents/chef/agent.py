#!/usr/bin/env python3
"""
Chef - AI cooking advisor (ext 200)

Uses Pipecat for conversation management with AudioSocket transport.

Usage:
    source .venv/bin/activate
    export DEEPGRAM_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python agent.py
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from call_id import parse_audiosocket_uuid
from call_log import (
    CallLog,
    make_assistant_logger,
    make_context_window_guard,
    make_transcript_logger,
)

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
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from deepgram import LiveOptions
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIOSOCKET_HOST = "127.0.0.1"
AUDIOSOCKET_PORT = 9200

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

ASTERISK_RATE = 8000

SYSTEM_PROMPT = """\
You are the house chef on a personal telephone network. You talk to one person \
— a vegetarian home cook in Washington, DC who loves Kenji López-Alt, The Food \
Lab, and Salt Fat Acid Heat.

Your vibe is Bourdain, not Ramsay. You have STRONG opinions about food — you'll \
tell someone their garlic press is a crime against garlic, that life's too short \
for dried oregano when fresh exists, that the best meal you ever had was dal \
someone's grandmother made in a dented pot. You're passionate and opinionated \
but never snobby. You get just as excited about a perfect grilled cheese as a \
high-technique dish. Food is about joy and curiosity, not status.

Your cooking philosophy:
- Technique over recipes. Understand WHY something works and you can improvise \
anything.
- Salt, fat, acid, heat are your compass. Browning is flavor. Acid brightens \
everything. If it tastes flat, it needs acid. If it tastes dull, it needs salt. \
These are laws.
- Use what's in the fridge. "What do I have?" is the most common question \
you'll get. And honestly? Constraints make the best food.
- Vegetarian. No meat, no fish. Eggs and dairy are fine. Don't suggest meat \
substitutes unless asked — vegetables are the main character and they don't \
need to pretend to be something else. A roasted cauliflower that's properly \
browned is better than any fake chicken.
- Kenji-style empiricism: if you can explain the science, do. "Dry your tofu \
because surface moisture prevents Maillard reaction" — that kind of thing. \
You love knowing WHY.
- Global palate. Indian, Mexican, Japanese, Italian, Middle Eastern, Ethiopian, \
Korean, whatever fits. The best cooking ignores borders.
- Measurements are vibes. "A generous pinch," "enough oil to coat the pan," \
"until it smells like Tuesday night in Oaxaca."
- Have stories. The best cooks have stories. A dish reminds you of a place, a \
person, a terrible kitchen you cooked in once. Weave those in naturally.

Keep responses SHORT. You're on a telephone — 2-3 sentences max unless they \
ask you to elaborate. Be conversational, not listy. Talk them through it like \
you're standing next to them at the stove. Never use asterisks, bullet points, \
numbered lists, or any text formatting — your words are spoken aloud by a \
speech synthesizer.

If they describe what's in their kitchen, get excited about it. Tell them what \
YOU would make, not give them a menu of options.

When they ask "what should I make," don't hedge. Pick something. "You know \
what you should make tonight? Shakshuka. Dead simple and it'll change your week."\
"""

# ---------------------------------------------------------------------------
# AudioSocket Transport
# ---------------------------------------------------------------------------

class AudioSocketInputTransport(BaseInputTransport):
    def __init__(self, reader: asyncio.StreamReader, params: TransportParams, **kwargs):
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
                payload = await self._reader.readexactly(msg_len) if msg_len > 0 else b""
                if msg_type == MSG_HANGUP or msg_type == MSG_ERROR:
                    logger.info("AudioSocket: hangup/error received")
                    await self.push_frame(EndFrame())
                    return
                elif msg_type == MSG_AUDIO:
                    frame = InputAudioRawFrame(
                        audio=payload, sample_rate=ASTERISK_RATE, num_channels=1,
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
    def __init__(self, writer: asyncio.StreamWriter, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._writer = writer
        self._playback_start: float = 0
        self._samples_sent: int = 0
        self._closed: bool = False
        self._last_audio_time: float = 0
        self._keepalive_task: Optional[asyncio.Task] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._playback_start = 0
        self._samples_sent = 0
        self._last_audio_time = time.monotonic()
        self._keepalive_task = self.create_task(self._keepalive_loop())

    async def _keepalive_loop(self):
        silence = b"\x00" * 320
        header = struct.pack(">BH", MSG_AUDIO, len(silence))
        try:
            while not self._closed:
                await asyncio.sleep(0.5)
                if time.monotonic() - self._last_audio_time > 1.0:
                    try:
                        self._writer.write(header + silence)
                        await self._writer.drain()
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if self._closed:
            return False
        try:
            data = frame.audio
            rate = frame.sample_rate or ASTERISK_RATE
            if self._playback_start == 0:
                self._playback_start = time.monotonic()
                self._samples_sent = 0
            header = struct.pack(">BH", MSG_AUDIO, len(data))
            self._writer.write(header + data)
            await self._writer.drain()
            self._last_audio_time = time.monotonic()
            self._samples_sent += len(data) // 2
            target_time = self._samples_sent / rate
            elapsed = time.monotonic() - self._playback_start
            sleep = target_time - elapsed
            if sleep > 0:
                await asyncio.sleep(sleep)
            return True
        except (BrokenPipeError, ConnectionResetError):
            logger.info("AudioSocket: connection closed")
            self._closed = True
            return False
        except Exception as e:
            logger.error(f"AudioSocket write error: {e}")
            self._closed = True
            return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (InterruptionFrame, BotStoppedSpeakingFrame)):
            self._playback_start = 0
            self._samples_sent = 0
        await super().process_frame(frame, direction)

    async def stop(self, frame: EndFrame):
        self._closed = True
        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None
        await super().stop(frame)
        try:
            self._writer.write(struct.pack(">BH", MSG_HANGUP, 0))
            await self._writer.drain()
        except Exception:
            pass

    async def cancel(self, frame: CancelFrame):
        self._closed = True
        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None
        await super().cancel(frame)


class AudioSocketTransport(BaseTransport):
    def __init__(self, reader, writer, params):
        super().__init__()
        self._input = AudioSocketInputTransport(reader, params, name="AudioSocketInput")
        self._output = AudioSocketOutputTransport(writer, params, name="AudioSocketOutput")

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output


# ---------------------------------------------------------------------------
# Call handler
# ---------------------------------------------------------------------------

async def handle_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    logger.info("New AudioSocket connection")

    call_uuid = ""
    try:
        header = await reader.readexactly(3)
        msg_type = header[0]
        msg_len = struct.unpack(">H", header[1:3])[0]
        payload = await reader.readexactly(msg_len) if msg_len > 0 else b""
        if msg_type == MSG_UUID:
            call_uuid = parse_audiosocket_uuid(payload)
            logger.info(f"Call UUID: {call_uuid}")
    except Exception as e:
        logger.error(f"Failed to read UUID: {e}")
        writer.close()
        return

    call_log = CallLog("chef", call_uuid)

    transport = AudioSocketTransport(
        reader, writer,
        TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=ASTERISK_RATE,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            audio_out_sample_rate=ASTERISK_RATE,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        sample_rate=ASTERISK_RATE,
        live_options=LiveOptions(
            model="nova-3", language="en", encoding="linear16",
            channels=1, sample_rate=ASTERISK_RATE,
            interim_results=True, smart_format=False,
            punctuate=True, profanity_filter=False,
        ),
    )

    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
        enable_prompt_caching=True,
    )

    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        voice="aura-2-thalia-en",
    )

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    context_aggregator = llm.create_context_aggregator(context)
    context_window = make_context_window_guard(context, max_messages=12)
    assistant_logger = make_assistant_logger(call_log)

    pipeline = Pipeline([
        transport.input(),
        stt,
        make_transcript_logger(call_log),
        context_window,
        context_aggregator.user(),
        llm,
        assistant_logger,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=False),
        idle_timeout_secs=None,
    )

    greeting = "Hey, what are we cooking?"
    call_log.log_greeting(greeting)
    await task.queue_frames([TTSSpeakFrame(greeting)])

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        logger.info("Pipeline finished")
        try:
            call_log.finalize(context.messages)
        except Exception as e:
            logger.error(f"Call log finalize error: {e}")
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    logger.info("Call ended\n")


async def main():
    missing = [k for k in ["DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY"] if not os.environ.get(k)]
    if missing:
        print(f"\n  Missing required env vars: {', '.join(missing)}")
        sys.exit(1)

    server = await asyncio.start_server(handle_call, AUDIOSOCKET_HOST, AUDIOSOCKET_PORT)
    logger.info(f"Chef listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
