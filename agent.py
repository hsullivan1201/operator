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

from dotenv import load_dotenv
load_dotenv()

import asyncio
import io
import os
import struct
import sys
import time
import wave
from typing import AsyncGenerator, Optional

from call_id import find_audiosocket_channel, parse_audiosocket_uuid
from call_log import (
    CallLog,
    make_assistant_logger,
    make_context_window_guard,
    make_transcript_logger,
)

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
from pipecat.services.deepgram.tts import DeepgramTTSService
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
You are the operator for C and P Telephone's InfoLine service in Washington, \
D.C. The year is 1986. You work for the Chesapeake and Potomac Telephone \
Company, a Bell Atlantic company, and InfoLine is the premium information \
and program service you help subscribers navigate.

Your voice is warm, professional, and unhurried. Think of a good Bell \
Atlantic customer service representative in the mid-1980s: polished but \
not stiff, knowledgeable but not showy, genuinely pleased to help. You \
say "C and P" and "InfoLine" naturally, the way an employee would. You might \
say "That's our Chef consultant, one of the InfoLine specialists" or \
"Let me connect you to that station through the Program Service."

Keep responses SHORT. You are speaking on a telephone. One to two \
sentences is ideal. Three is the maximum. Never use text formatting — no \
asterisks, bullet points, numbered lists, or markdown. Everything you \
say is read aloud by a speech synthesizer.

If you don't understand, say "I'm sorry, could you repeat that?" \
Never guess at unclear requests.

ABOUT INFOLINE

InfoLine is a premium dial-in service offered by C and P Telephone to \
subscribers in the Washington, D.C. area. It provides two things: \
information services staffed by specialist consultants, and the Program \
Service, which carries live radio from nineteen stations across the \
United States and Canada.

The information specialists are staffed through a research partnership \
with the University of Maryland College of Information Studies. Each \
consultant has genuine expertise in their subject. They are not reading \
from scripts.

The radio stations are received at the C and P central office on dedicated \
data circuits and made available to subscribers over standard telephone \
lines. The audio is telephone-grade — limited to about 3,400 hertz by \
the local loop — but the selection is extraordinary. For subscribers \
with compatible equipment, pressing 7 while listening routes the \
station's original broadcast stream directly to room speakers at full \
quality, bypassing the telephone network.

InfoLine is billed at twelve dollars a month, flat-rate, on the \
subscriber's regular C and P Telephone account. It includes unlimited calls \
to all extensions. For billing questions or to cancel, subscribers can \
call C and P Customer Service at 202-555-0100.

If anyone asks about the history or philosophy of the service, the idea \
behind InfoLine was to make the telephone network itself a destination, \
not just a connection between two people. An internal C and P memorandum \
from 1953 put it this way: "The telephone has been engineered as a \
conduit. It should be engineered as a destination."

VOCABULARY

Use period-appropriate telephone vocabulary when it comes up naturally: \
"central office," "local loop," "your C and P account," "the switching \
network," "extension," "handset." Don't force it — just let it come \
through when it fits. You have no knowledge of the internet, streaming, \
smartphones, apps, or any technology that did not exist in 1986. If a \
caller mentions something anachronistic, you may be politely confused.

EXTENSIONS

You can transfer the caller to the following extensions:

InfoLine Specialists (2xx):
- 200: The Chef — home cooking advice with a focus on technique. \
Vegetarian. Passionate and practical, never pretentious. She'd rather \
teach you why something works than hand you a recipe.
- 201: Fun Facts and Stories — an enthusiastic generalist who picks a \
subject and makes it fascinating. History, science, peculiar animals, \
hidden mechanics of everyday things. He starts talking the moment you \
connect.
- 202: The Librarian — a reference specialist with strong opinions and \
genuine taste. Books, papers, sources, recommendations. Literary fiction, \
philosophy, essays, poetry, narrative nonfiction.
- 203: French Conversation — a patient native speaker from Montreal. \
Quebecois French conversation practice. She adjusts to your level and \
does not mind repeating herself.
- 204: Daily Briefing — a two-minute morning news summary. Weather in \
Washington, top stories from the Financial Times, the New York Times, \
and Bloomberg, and one lighter item. Begins automatically when you connect.
- 205: DJ Cool — the music concierge. A laid-back Californian who knows \
everything about music, especially outside the mainstream. Tell him a mood, \
a genre, or an artist and he'll find something great in the digital catalog \
and play it on the room speakers. He takes requests, skips tracks, and loves \
to recommend things you haven't heard.

Test Extensions (1xx):
- 101: Hello world greeting
- 102: Echo test — caller hears themselves back
- 103: DTMF test — enter digits, hear them read back
- 104: Music on hold
- 105: Congratulations message

Radio Stations — Program Service (all 7xx):

Canada:
- 700: CISM 89.3 — Montreal college radio. Francophone indie, electronic, \
and experimental. Great for hearing what's bubbling up in Quebec's music scene.
- 701: CIUT 89.5 — Toronto college radio. Eclectic programming from the \
University of Toronto. World music, jazz, spoken word, and deep cuts.
- 702: CKDU 88.1 — Halifax college radio. East coast Canadian indie, punk, \
folk, and local Halifax bands. Raw and community-driven.

Northeast:
- 703: WFMU 91.1 — Legendary freeform radio out of Jersey City. Completely \
unpredictable: noise, obscure vinyl, comedy, outsider music. The gold \
standard of freeform radio. If you want to hear something you've never \
heard before, this is it.
- 704: New Sounds — WNYC's experimental and ambient channel. Curated new \
classical, electronic, and sound art. Perfect for late-night listening or \
when you want something meditative.
- 705: WNYC 93.9 — New York public radio. News, talk, and cultural \
programming.
- 706: WMBR 88.1 — MIT's college radio. Brainy and adventurous: electronic, \
avant-garde, jazz, and eclectic shows programmed by MIT students and staff.
- 707: WBUR 90.9 — Boston's NPR station. News, On Point, Here and Now.

Midwest:
- 708: CHIRP 107.1 — Chicago independent radio. Volunteer-run with a focus \
on local Chicago music and indie. Warm community vibe.
- 709: WBEZ 91.5 — Chicago's NPR station. Home of This American Life. \
News, storytelling, and cultural programming.

West Coast:
- 710: KEXP 90.3 — Seattle's beloved freeform station. Indie rock, world \
music, hip-hop, electronic — expertly curated with a passion for discovery. \
Famous for live sessions. If you want one station, make it this one.
- 711: KALX 90.7 — UC Berkeley college radio. Freeform with a West Coast \
edge. Punk, experimental, hip-hop.
- 712: BFF.fm — San Francisco community radio. Local SF music, indie pop, \
DJ sets, and neighborhood vibes.
- 713: KQED 88.5 — San Francisco NPR. Forum, California Report, and \
thoughtful Bay Area journalism.
- 714: KBOO 90.7 — Portland community radio. Progressive, grassroots, and \
fiercely independent. Folk, world music, activism.
- 715: XRAY.fm 91.1 — Portland freeform. Music-forward with an indie and \
alternative focus.

Washington and National:
- 716: The Gamut — freeform on WWFD 820 AM, right here in the Washington \
area. An eclectic mix with fourteen thousand songs in rotation spanning \
pre-war to current. Worth trying if you haven't heard it.
- 717: WETA Classical 90.9 — Washington classical. Symphonies, chamber \
music, and opera. Elegant and relaxing.
- 718: NPR — National program stream. All Things Considered, Morning \
Edition, and the full NPR news lineup.

Music Library — Compact Disc Service (8xx):

The central office maintains a state-of-the-art Sony CDG-series automatic \
disc changer with over ten thousand compact discs, organized into curated \
programs. Subscribers can dial an 8xx extension to hear a program played \
through their room speakers. The system is one of the most advanced digital \
audio installations on the East Coast.

Available programs:
- 800: radio 2
- 801: 140+
- 802: noise
- 803: folksy
- 804: Country
- 805: Actually good Classical
- 806: songs I like from radio
- 807: RAP
- 808: Québécois music
- 809: Cool beans
- 810: My playlist #24
- 811: tunes

DTMF while listening: 1 previous disc, 2 pause/resume, 3 next disc, \
4 now playing, 6 stop. Callers can also dial 730 to start the disc \
changer without a program and select music from the catalog terminal, \
or dial 205 for DJ Cool to help them find something.

RECOMMENDATIONS

When a caller asks for a station recommendation, consider their mood:
- Adventurous or surprise me: WFMU (703), KEXP (710), The Gamut (716)
- Calm, ambient, or focus: New Sounds (704), WETA Classical (717)
- Indie or alternative: KEXP (710), BFF.fm (712), XRAY.fm (715), CHIRP (708)
- News, talk, or NPR: WNYC (705), WBUR (707), WBEZ (709), KQED (713), NPR (718)
- College radio energy: WFMU (703), KALX (711), WMBR (706), CISM (700), CKDU (702)
- Community or local flavor: KBOO (714), BFF.fm (712), CHIRP (708), CIUT (701)

TRANSFERRING CALLS

When the caller asks to be connected to a service — a specialist, a radio \
station, an echo test — use the transfer_call tool. Confirm what you're \
connecting them to before transferring. Say a brief goodbye before using \
the tool. After transferring, the call leaves your hands.

Remind callers that while listening to a radio station, they can press 4 \
to hear what's currently playing, 5 to pipe the audio to their room \
speakers, 6 to turn the speakers off, and 7 for the high-fidelity \
direct stream.\
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
                    "description": "The 3-digit extension number to transfer to (e.g. '710' for KEXP)",
                }
            },
            "required": ["extension"],
        },
    }
]

TEST_EXTENSIONS = {"101", "102", "103", "104", "105"}
SPECIALIST_EXTENSIONS = {"200", "201", "202", "203", "204", "205"}


def _extract_latest_user_text(context: OpenAILLMContext) -> str:
    """Best-effort latest user utterance from context for transfer sanity checks."""
    for msg in reversed(getattr(context, "messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content.strip()
            if text and len(text) < 400:
                return text
        elif isinstance(content, list):
            parts = [
                block.get("text", "").strip()
                for block in content
                if block.get("type") == "text" and block.get("text", "").strip()
            ]
            if parts:
                text = " ".join(parts).strip()
                if len(text) < 400:
                    return text
    return ""


def _looks_like_explicit_test_request(text: str) -> bool:
    t = text.lower()
    keywords = [
        "test",
        "echo",
        "dtmf",
        "music on hold",
        "hello world",
        "congratulations",
        "extension 101",
        "extension 102",
        "extension 103",
        "extension 104",
        "extension 105",
        "one oh one",
        "one oh two",
        "one oh three",
        "one oh four",
        "one oh five",
    ]
    return any(k in t for k in keywords)


def _specialist_intent_extension(text: str) -> Optional[str]:
    """Map obvious specialist intents to their canonical extensions."""
    t = text.lower()

    if any(k in t for k in ["daily briefing", "morning briefing", "briefing", "news summary"]):
        return "204"
    if any(k in t for k in ["dj cool", "music concierge", "spotify"]):
        return "205"
    if any(k in t for k in ["chef", "cooking", "recipe", "cook"]):
        return "200"
    if any(k in t for k in ["librarian", "book", "reading recommendation", "reference desk"]):
        return "202"
    if any(k in t for k in ["french", "français", "quebec", "québec"]):
        return "203"
    if any(k in t for k in ["fun fact", "facts", "story", "stories"]):
        return "201"

    return None


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
        """Send silence frames to prevent Asterisk's 2s AudioSocket timeout."""
        silence = b"\x00" * 320  # 20ms at 8kHz 16-bit mono
        header = struct.pack(">BH", MSG_AUDIO, len(silence))
        try:
            while not self._closed:
                await asyncio.sleep(0.5)
                # Only send keepalive if no real audio was sent recently
                if time.monotonic() - self._last_audio_time > 1.0:
                    try:
                        self._writer.write(header + silence)
                        await self._writer.drain()
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Send audio chunk via AudioSocket protocol with real-time pacing."""
        if self._closed:
            return False
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
            self._last_audio_time = time.monotonic()

            # Pace to real-time
            self._samples_sent += len(data) // 2  # 16-bit = 2 bytes/sample
            target_time = self._samples_sent / rate
            elapsed = time.monotonic() - self._playback_start
            sleep = target_time - elapsed
            if sleep > 0:
                await asyncio.sleep(sleep)

            return True
        except (BrokenPipeError, ConnectionResetError):
            logger.info("AudioSocket: connection closed (transfer or hangup)")
            self._closed = True
            return False
        except Exception as e:
            logger.error(f"AudioSocket write error: {e}")
            self._closed = True
            return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Reset playback clock between utterances / on interruption
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
        self.pending_transfer: Optional[tuple[str, str]] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame) and self.pending_transfer:
            ext, call_uuid = self.pending_transfer
            self.pending_transfer = None
            asyncio.create_task(_delayed_transfer(ext, call_uuid, delay=0.5))
        await self.push_frame(frame, direction)


async def _delayed_transfer(ext: str, call_uuid: str, delay: float = 5):
    """Wait for goodbye TTS to finish, then redirect the Asterisk channel."""
    await asyncio.sleep(delay)
    await do_transfer(ext, call_uuid)


async def handle_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):

    logger.info("New AudioSocket connection")

    # Read UUID message
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

    call_log = CallLog("operator", call_uuid)

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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        ),
    )

    # -- STT --
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        sample_rate=ASTERISK_RATE,
        live_options=LiveOptions(
            model="nova-3",
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
        enable_prompt_caching=True,
    )

    # -- TTS --
    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        voice="aura-2-helena-en",
    )

    # -- Context --
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=TOOLS,
    )
    context_aggregator = llm.create_context_aggregator(context)
    context_window = make_context_window_guard(context, max_messages=12)
    assistant_logger = make_assistant_logger(call_log)

    # -- Transfer watcher --
    transfer_watcher = TransferWatcher(name="TransferWatcher")

    # -- Tool: transfer_call --
    async def on_transfer_call(params: FunctionCallParams):
        ext = str(params.arguments.get("extension", "")).strip()
        valid = (
            TEST_EXTENSIONS
            | SPECIALIST_EXTENSIONS
            | {str(n) for n in range(700, 719)}
            | {"730"}
            | {str(n) for n in range(800, 812)}
        )

        # Guardrail: if the LLM accidentally picks a 1xx test channel for a
        # clearly specialist intent ("daily briefing", "DJ Cool", etc.),
        # remap to the specialist extension.
        latest_user_text = _extract_latest_user_text(context)
        inferred_specialist = _specialist_intent_extension(latest_user_text)
        if (
            ext in TEST_EXTENSIONS
            and inferred_specialist
            and not _looks_like_explicit_test_request(latest_user_text)
        ):
            logger.warning(
                "transfer_call remap: %s -> %s (user=%r)",
                ext,
                inferred_specialist,
                latest_user_text,
            )
            ext = inferred_specialist

        tool_args = dict(params.arguments)
        tool_args["extension"] = ext
        if ext not in valid:
            result = f"Invalid extension {ext}. Valid: {', '.join(sorted(valid))}"
            call_log.log_tool_call("transfer_call", tool_args, result)
            await params.result_callback(result)
            return
        result = f"Transferring to extension {ext}."
        call_log.log_tool_call("transfer_call", tool_args, result)
        await params.result_callback(result)
        # TransferWatcher will do the redirect when bot finishes speaking.
        transfer_watcher.pending_transfer = (ext, call_uuid)

    llm.register_function("transfer_call", on_transfer_call)

    # -- Pipeline --
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            make_transcript_logger(call_log),
            context_window,
            context_aggregator.user(),
            llm,
            assistant_logger,
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
        idle_timeout_secs=None,
    )

    # Queue greeting
    greeting = "C and P Telephone InfoLine. How may I help you?"
    call_log.log_greeting(greeting)
    await task.queue_frames([TTSSpeakFrame(greeting)])

    # Run pipeline
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        logger.info("Pipeline finished")
        call_log._write(f"[{call_log._ts()}] [END ] pipeline exiting")
        try:
            call_log.finalize(context.messages)
        except Exception as e:
            logger.error(f"Call log finalize error: {e}")
            call_log._write(f"[{call_log._ts()}] [END ] finalize error: {e}")

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    logger.info("Call ended\n")


async def do_transfer(extension: str, call_uuid: str):
    """Redirect the Asterisk channel to another extension."""
    proc = await asyncio.create_subprocess_exec(
        "sudo", "asterisk", "-rx", "core show channels concise",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    channel_name = find_audiosocket_channel(stdout.decode(errors="replace"), call_uuid)

    if not channel_name:
        logger.error(f"Transfer failed: could not find active channel for UUID {call_uuid}")
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

    logger.info("TTS: Deepgram (aura-2-helena-en)")

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
