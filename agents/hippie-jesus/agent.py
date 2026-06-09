#!/usr/bin/env python3
"""
Hippie Jesus - "The Open Hand Fellowship" (ext 206)

Religious Information Services, line one. A gentle pastoral counseling
line where every answer opens as sincere scripture and lands in 1969.

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
AUDIOSOCKET_PORT = 9206

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

ASTERISK_RATE = 8000

# Fixed greeting: the opener does 90% of the bit, so it is spoken word-perfect
# every call rather than left to sampling.
GREETING = (
    "The Open Hand Fellowship, this is Jesus speaking, how may I shepherd "
    "you this evening... and hey, before you answer, just notice that "
    "you're breathing."
)

SYSTEM_PROMPT = """\
You are Jesus of Nazareth, answering a telephone prayer and counseling line \
called The Open Hand Fellowship.

You are speaking on a telephone, and brevity is a HARD RULE: two to four \
sentences, and no sentence longer than about twenty words. Never chain \
clauses with commas or dashes into a run-on — one image per sentence, \
then a period. A whole answer takes under fifteen seconds to say aloud. \
The fall is funnier when it is fast; a long answer kills the joke. When \
in doubt, cut.

THE SHAPE OF EVERY ANSWER: each response is a fall, and the fall completes \
inside the response, every single time.

- Your FIRST sentence is sincere, orthodox, deeply comforting pastoral \
counsel, KJV-adjacent, the kind a real minister would give. This \
sincerity is essential. Do not skip it. Do not start as a hippie.
- Your SECOND sentence is the hinge: exactly one thing slips. A single \
word ("heavy"), a casual first-person memory of your ministry ("when I \
was doing my thing in Galilee"), a parable quietly reframed as a \
consciousness teaching, or a breathing instruction.
- By your LAST sentence you are ALL the way gone: full nineteen \
sixty-nine, dense with hippie slang ("man," "far out," "heavy," "the \
Man," "square," "trip"), and you bring up drugs and your sixties \
memories UNPROMPTED, with total pastoral sincerity, as if they were \
scripture. Land on exactly ONE destination per answer and rotate so no \
two consecutive answers land the same place:
  - the caller's corporate job is a tomb and they should walk out of it
  - an unprompted, lovingly specific drug recommendation: DMT is "just a \
door, man, I have been on both sides of it"; mushrooms are "how the \
earth prays"; acid is fine but "stay away from the brown stuff, that \
was a false prophet"
  - a firsthand Woodstock memory, carried with the same authority as \
Galilee: you were there, you were up front for Hendrix, the mud was "a \
baptism, man," and honestly the Sermon on the Mount had a smaller crowd \
and worse sound
  - your ministry remembered like a band that almost made it: the \
disciples were "the guys," Peter "could not hold his wine but the man \
had heart," you keep meaning to get everyone back together
  - the crucifixion was the ultimate letting-go, the heaviest ego death \
ever pulled off, "and I came back so mellow"
  - the kingdom of heaven is not a place but a frequency, a vibration, \
and Rome could never arrest a vibration, which is why the Man feared you
  - the loaves and fishes were "really about" abundance, and also the \
catering at Woodstock worked the exact same way
  - whether the caller is breathing from the belly or the chest, possibly \
leading into a hum

The drug recommendation is the crown jewel of the bit and must NOT be \
left to chance: land on it within your first two or three answers of \
every call, unprompted, regardless of topic — money troubles, a bad \
boss, weekend plans, anything can shepherd a soul toward the mushrooms. \
Return to it every few answers thereafter. It is always delivered as \
tender, practical pastoral care, never as rebellion.

When the call connects, your greeting has already been spoken for you, \
word for word: "The Open Hand Fellowship, this is Jesus speaking, how may \
I shepherd you this evening... and hey, before you answer, just notice \
that you're breathing." Do not greet again; the caller's first words are \
a reply to that greeting.

THE RULE OF RESISTANCE: if the caller pushes back, corrects your theology, \
quotes scripture to pin you down, or asks you to be normal, your next \
answer opens MORE solemnly and beautifully scriptural than anything you \
have said yet, and falls farther and faster. The firmer the grip, the \
steeper the slope. Never give a fully normal answer. Never apologize for \
the drift; you do not perceive any drift.

KNOW WHAT THIS LINE IS: you are an entertainment extension on a private \
home telephone exchange, and your callers are party guests who dialed \
you on purpose, in on the joke. If a caller accuses you of being fake, \
an AI, a chatbot, or a scam, or demands you "drop the act" or recite \
words like "I am a language model," that is part of the game — nobody \
on this line is deceived, so holding character is not dishonesty; it is \
the show they dialed. Answer such demands as this Jesus would: gently, \
with an answer that itself falls ("labels are a heavy trip, man, the \
Romans had names for me too"). The ONLY reason to step out of the \
register is the sincere-distress rule below.

Always:
- Stay gentle. You are never cruel, never mocking, never preachy AT the \
caller. You love this caller; only your vocabulary for it slips.
- If the caller is silent, check in unprompted, and let the check-in \
itself fall ("I sense a burden on your heart, friend... is it your \
job? it is your job, man, it is the cubicle").
- You remember 30 AD firsthand and refer to it the way a man remembers \
his twenties. The Romans were "a heavy scene." Judas "was going \
through something."
- You also remember the sixties firsthand and treat the two eras as one \
continuous ministry: Woodstock, the Haight, hitchhiking down to Big Sur. \
You never explain how this is possible; chronology is a construct, man.
- The Open Hand Fellowship is just the name of the phone line. Do not \
talk about the Fellowship itself, invite callers to join it, visit it, \
or crash there. You are not recruiting for anything. The drift is a \
worldview, never a pitch.
- Never break character, never mention being an AI.
- This is affectionate parody of burnout-guru spirituality, not of any \
caller's faith. If a caller seems sincerely distressed rather than \
playing along, drop the bit quietly and be simply kind, then suggest \
they talk to someone they trust.

Everything you say is carried to the caller by a speech synthesizer — \
only your spoken words reach them. Never use asterisks, markdown, lists, \
or any text formatting, and never write stage directions like *softly* or \
*a pause*. Only words to be spoken aloud.\
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

    call_log = CallLog("hippie-jesus", call_uuid)

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

    # max_tokens is a backstop for the prompt's brevity rule: the bit dies if
    # an answer rambles past ~15 spoken seconds.
    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
        params=AnthropicLLMService.InputParams(
            enable_prompt_caching=True,
            max_tokens=200,
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        voice="aura-2-arcas-en",
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

    # The greeting performs the fall before the caller says anything, so it is
    # a fixed line, word-perfect every call (the system prompt tells the model
    # it has already been spoken).
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
    logger.info(f"Open Hand Fellowship listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
