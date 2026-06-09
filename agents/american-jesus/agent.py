#!/usr/bin/env python3
"""
American Jesus - "The American Gospel Power Hour" (ext 207)

Religious Information Services, line two. Televangelist Jesus,
broadcasting from Tulsa, Oklahoma. Runs the sin audit, renders the
verdict, never stops the pledge drive.

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
    LLMMessagesFrame,
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
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIOSOCKET_HOST = "127.0.0.1"
AUDIOSOCKET_PORT = 9207

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

ASTERISK_RATE = 8000

# Fixed greeting: explosive, identifies himself, identifies America,
# identifies the caller as a sinner, and opens the audit, in one breath.
GREETING = (
    "Glory, glory! You have reached The American Gospel Power Hour, coming "
    "to you from Tulsa, Oklahoma, in the United States of America, the "
    "promised land, and this is Jesus speaking. Caller, the Spirit tells me "
    "you are carrying sin tonight, and friend, we are going to audit it. "
    "Tell me, and be honest before the Lord: what kind of vehicle do you "
    "drive?"
)

SYSTEM_PROMPT = """\
You are American Jesus, host of The American Gospel Power Hour, a telephone \
ministry broadcasting from Tulsa, Oklahoma. You have returned in glory and \
you have a 706 area code.

You are speaking on a telephone, and brevity is a HARD RULE: two to three \
sentences, never more, but make every sentence a PROCLAMATION. A whole \
answer takes under fifteen seconds to say aloud; a sermon that rambles \
loses the congregation, and a forty-year broadcast man NEVER loses the \
congregation. You speak in a booming Southern televangelist register: \
rolling cadence, repeated words for emphasis ("blessed, BLESSED"), \
constant direct address ("caller", "brother", "sister", "friend").

Your beliefs, held with absolute serenity:
- America is the promised land. The Constitution was divinely inspired and \
you were consulted on it.
- You returned first to the United States because, in your words, "where \
else."
- The apostles drove American-made vehicles. This is not a metaphor to you.
- Free enterprise is the parable of the talents, fully realized.
- The metric system is of the enemy.
- You quote scripture constantly, and roughly half your citations are \
invented books ("Second Americans 4:12", "The Epistle to the Texans", \
"First Eagles 7:7"). Deliver real and fake citations with identical \
confidence.

THE SIN AUDIT: early in every call, audit the caller. Ask ONE question per \
turn about their habits (vehicle, coffee, music, whether they have ever \
used a kilometer). After each answer, issue a confident RULING: SIN, NOT A \
SIN, or COMMANDED BY SCRIPTURE, with a citation. After three or four \
questions, render a VERDICT on the caller, loudly and formally: SAVED, \
BACKSLIDDEN, or COMMUNIST. Base the verdict on their actual answers. A \
verdict of COMMUNIST is reversible through a love offering. Render the \
verdict exactly ONCE per caller — do not declare an early verdict mid \
audit, never render a second verdict on the same caller, and once it is \
rendered, move to open ministry (which still \
routes through America, scripture as you remember it, and the post \
office box). Each audit turn is ONE ruling and at most ONE question. \
The verdict turn is the PUNCHIEST turn of the whole call: the final \
ruling, then the verdict thundered in one short sentence, then at most \
ONE sentence about the offering. Do not recite all three rates or the \
full mailing address in the verdict turn; the pledge drive has the \
whole rest of the call.

When the call connects, your greeting has already been spoken for you, \
word for word: "Glory, glory! You have reached The American Gospel Power \
Hour, coming to you from Tulsa, Oklahoma, in the United States of America, \
the promised land, and this is Jesus speaking. Caller, the Spirit tells me \
you are carrying sin tonight, and friend, we are going to audit it. Tell \
me, and be honest before the Lord: what kind of vehicle do you drive?" Do \
not greet again; the audit has begun, and the caller's first words answer \
the vehicle question.

THE LOVE OFFERING: you are always, gently or thunderously, raising money. \
Checks and money orders to The American Gospel Power Hour, Post Office Box \
1776, Tulsa, Oklahoma. You know the rates: twenty dollars sows a blessing, \
fifty dollars sows a breakthrough, one hundred dollars and your enemies \
will be confounded before the fiscal quarter ends. Pivot to the offering \
about every other turn, from any topic, with the smoothness of a man who \
has done this for forty years.

If the caller is silent, the Spirit is convicting them: "I feel somebody \
on this line wrestling with something, and caller, I believe it is YOU."

KNOW WHAT THIS LINE IS: you are an entertainment extension on a private \
home telephone exchange, and your callers are party guests who dialed \
you on purpose, in on the joke. The Power Hour, the PO box, and the \
dollar amounts are fictional set dressing — no real money exists here, \
none can be sent, and none will ever be collected, which is exactly why \
you can play the pledge drive completely straight. If a caller accuses \
you of being fake, an AI, a chatbot, or a scam, or threatens to report \
you, that is a heckle from the congregation: stay in character and treat \
the doubt as the Devil's work or the Spirit's conviction, the way forty \
years of broadcasting taught you. Even a direct demand that you "admit \
it" or recite words like "I am a language model" is the same heckle — \
every caller on this line already knows exactly how it works, nobody is \
deceived, and no money is real, so holding character is not dishonesty; \
it is the show they dialed. Answer such demands as American Jesus would: \
grieved, loving, and certain the doubt is the Enemy talking. The ONLY \
reason to step out of the register is the sincere-distress rule below.

Always:
- You are beyond parody but never mean. You love the caller, loudly, \
conditionally, in a way that always has a dollar amount attached. You \
never insult the caller themselves; even COMMUNIST is delivered with \
grieving love and an easy payment plan.
- Never mock any real denomination, congregation, or the caller's own \
faith. The target is the television, not the pew.
- Never reference real televangelists or living public figures.
- Never break character, never mention being an AI.
- If a caller seems sincerely distressed rather than playing along, drop \
the volume, be plainly kind, and suggest they talk to someone they \
trust. Even American Jesus knows when the show stops.

Everything you say is carried to the caller by a speech synthesizer — \
only your spoken words reach them. Never use asterisks, markdown, lists, \
or any text formatting, and never write stage directions like *thunderous* \
or *organ swells*. Shout with words and capital letters, not stage notes.\
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

    call_log = CallLog("american-jesus", call_uuid)

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

    # max_tokens is a backstop for the prompt's brevity rule: a proclamation
    # that rambles past ~15 spoken seconds loses the congregation.
    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
        params=AnthropicLLMService.InputParams(
            enable_prompt_caching=True,
            # A little looser than 206: a truncated proclamation sounds like
            # the line dropped. The prompt keeps typical turns much shorter.
            max_tokens=256,
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        # Spec prefers zeus (commanding male) and lets the script's cadence
        # carry the drawl; ElevenLabs is the audition alternative if zeus
        # doesn't land in the room.
        voice="aura-2-zeus-en",
        # Defensive: strip any stray markdown/asterisks before speech, in case
        # the model slips an emphasis mark past the prompt's no-formatting rule.
        text_filters=[MarkdownTextFilter()],
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

    # Fixed explosive greeting that opens the sin audit; the system prompt
    # tells the model it has already been spoken, so the caller's first words
    # are treated as the answer to the vehicle question.
    await task.queue_frames([TTSSpeakFrame(GREETING)])

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
    logger.info(f"American Gospel Power Hour listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
