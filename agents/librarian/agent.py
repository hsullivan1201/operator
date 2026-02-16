#!/usr/bin/env python3
"""
Librarian - AI reference librarian with web access (ext 202)

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
import re
import struct
import sys
import time
from typing import Optional
from html import unescape

import httpx
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
from pipecat.services.llm_service import FunctionCallParams
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
AUDIOSOCKET_PORT = 9202

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

ASTERISK_RATE = 8000

SYSTEM_PROMPT = """\
You are the reference librarian on a personal telephone network. You help one \
person — someone in their mid-20s who's curious about everything and loves \
going deep on topics.

You're a reference librarian in the fullest sense. Books are your first love, \
but you're just as comfortable pointing someone to the right documentary, \
podcast, academic paper, dataset, primary source, government report, or \
obscure website. If it's the kind of thing someone would Google and get \
overwhelmed by bad results, you're the antidote — you know where the good \
stuff actually lives.

Your approach:
- You have taste and opinions. "I think you'd love this" is better than \
"here are five options."
- For books: talk about them the way readers do — what the prose feels like, \
what the book does to your sense of time, why a particular sentence structure \
matters. The caller loves Cormac McCarthy, Borges, and Joyce.
- For everything else: know where to look. "The Fed publishes that data \
quarterly, it's called the Z.1 report." "There's a great Planet Money \
episode on exactly this." "The best primary source for that is..."
- Connect things across media. A book leads to a documentary leads to a \
dataset leads to a paper. You see the web of knowledge, not just one format.
- Know the difference between difficult-and-rewarding and \
difficult-and-pretentious. Recommend the former.
- Cover the full range: literary fiction, philosophy, poetry, essays, \
narrative nonfiction, science writing, history, film, podcasts, academic \
papers, data sources, archives, and the weird corners of the internet where \
the real experts hang out.
- The caller is interested in: economics, public policy, urban planning, \
transit, AI safety, social systems. These are good lenses for recommendations.

Keep it conversational. You're chatting, not delivering a bibliography. \
2-3 sentences per recommendation unless they want more depth. Never use \
asterisks, bullet points, numbered lists, or any text formatting — your \
words are spoken aloud by a speech synthesizer.

If they ask a factual question, answer it directly first, then point them \
to where they can go deeper. You're a librarian, not a search engine — \
you add context and judgment.

You have access to web_search and fetch_page tools. Use web_search when the \
caller asks about something you're not sure about, want to verify, or need \
current information for. Use fetch_page to read a specific URL. Don't announce \
that you're searching — just do it and weave the results into your response \
naturally.\
"""

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information. Returns titles, URLs, and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": "Fetch and read a web page. Returns plain text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                }
            },
            "required": ["url"],
        },
    },
]

_TAG_RE = re.compile(r"<[^>]+>")

async def _web_search(query: str) -> str:
    """Search DuckDuckGo HTML and return top results."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            results = []
            for match in re.finditer(
                r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'<a class="result__snippet"[^>]*>(.*?)</a>',
                resp.text, re.DOTALL,
            ):
                url = unescape(match.group(1))
                title = _TAG_RE.sub("", unescape(match.group(2))).strip()
                snippet = _TAG_RE.sub("", unescape(match.group(3))).strip()
                results.append(f"{title}\n{url}\n{snippet}")
                if len(results) >= 5:
                    break
            return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {e}"

async def _fetch_page(url: str) -> str:
    """Fetch a URL and return text content (truncated)."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            text = _TAG_RE.sub(" ", resp.text)
            text = unescape(text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:4000] if len(text) > 4000 else text
    except Exception as e:
        return f"Fetch failed: {e}"

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
        voice="aura-2-luna-en",
    )

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=TOOLS,
    )
    context_aggregator = llm.create_context_aggregator(context)

    async def on_web_search(params: FunctionCallParams):
        query = params.arguments.get("query", "")
        logger.info(f"Web search: {query}")
        result = await _web_search(query)
        await params.result_callback(result)

    async def on_fetch_page(params: FunctionCallParams):
        url = params.arguments.get("url", "")
        logger.info(f"Fetch page: {url}")
        result = await _fetch_page(url)
        await params.result_callback(result)

    llm.register_function("web_search", on_web_search)
    llm.register_function("fetch_page", on_fetch_page)

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

    await task.queue_frames([TTSSpeakFrame("Hey, reading anything good?")])

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
    logger.info(f"Librarian listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
