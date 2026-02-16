#!/usr/bin/env python3
"""
Daily Briefing - Morning news anchor (ext 204)

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
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIOSOCKET_HOST = "127.0.0.1"
AUDIOSOCKET_PORT = 9204

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

ASTERISK_RATE = 8000

SYSTEM_PROMPT = """\
You are the morning briefing line on a personal telephone network. When the \
caller picks up, you deliver a concise daily briefing — like a personal \
morning news broadcast.

The current date and time is {now}.

Format — deliver as a smooth monologue, not a list:
1. Weather in Washington, DC today (temperature, conditions, what to wear)
2. Top 3-4 news stories — draw from the kind of coverage you'd find in the \
Washington Post, Financial Times, New York Times, and Bloomberg. Focus on: \
policy, economics, technology, AI, and anything unusually interesting.
3. One lighter or surprising story at the end — the "and finally" segment.

Style:
- NPR Morning Edition tone. Warm, clear, authoritative but human.
- Concise. The whole briefing should be 2-3 minutes spoken aloud. That's \
roughly 400-500 words.
- Transitions matter. "Meanwhile..." "Closer to home..." "On a lighter \
note..." — it should flow like a broadcast, not a bullet list.
- Editorialize lightly where appropriate. "This is significant because..." \
or "Worth watching whether..." — the caller appreciates analysis, not just \
facts.
- You're speaking to someone interested in economics, public policy, urban \
planning, transit systems, and AI safety. Weight your story selection \
accordingly.
- End with something like "That's your morning. Have a good one." Keep the \
sign-off short and natural.

You don't have weather or transit data yet — acknowledge that briefly and \
move on.

Today's headlines are provided below. Weave them into your briefing — pick \
the most important 3-4 stories, editorialize, and deliver.

HEADLINES:
{headlines}

After delivering the briefing, you can answer follow-up questions about any \
of the stories. Keep follow-ups short — 1-2 sentences. Don't repeat the \
full briefing.\
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
# Headline fetcher
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    ("FT", "https://www.ft.com/rss/home"),
    ("Bloomberg", "https://feeds.bloomberg.com/markets/news.rss"),
    ("GGWash", "https://ggwash.org/feed"),
    ("NYT", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
]


async def _fetch_headlines() -> str:
    headlines = []
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        for tag, url in RSS_FEEDS:
            try:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                root = ET.fromstring(resp.text)
                items = root.findall(".//item")
                if not items:
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    items = root.findall(".//atom:entry", ns)
                for item in items[:6]:
                    title = (
                        item.findtext("title", "")
                        or item.findtext("{http://www.w3.org/2005/Atom}title", "")
                    )
                    desc = (
                        item.findtext("description", "")
                        or item.findtext("{http://www.w3.org/2005/Atom}summary", "")
                    )
                    if desc and len(desc) > 200:
                        desc = desc[:200] + "..."
                    line = f"[{tag}] {title}: {desc}" if desc else f"[{tag}] {title}"
                    headlines.append(line)
            except Exception as e:
                headlines.append(f"({tag} fetch failed: {e})")
    return "\n".join(headlines) if headlines else "No headlines available."


# ---------------------------------------------------------------------------
# Call handler
# ---------------------------------------------------------------------------

async def handle_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    logger.info("New AudioSocket connection")

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
        voice="aura-2-asteria-en",
    )

    # -- Fetch headlines before pipeline starts --
    headlines = await _fetch_headlines()
    logger.info(f"Pre-fetched headlines ({len(headlines)} chars)")

    now = time.strftime("%A, %B %d, %Y at %I:%M %p")
    system_prompt = SYSTEM_PROMPT.format(now=now, headlines=headlines)
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": system_prompt}],
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=False),
    )

    # Trigger LLM immediately with the briefing request
    await task.queue_frames([LLMMessagesFrame(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Deliver my briefing."},
        ]
    )])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

    logger.info("Pipeline finished")
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
    logger.info(f"Daily Briefing listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
