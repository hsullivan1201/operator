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
import html
import os
import re
import struct
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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

You will receive a structured briefing packet with:
- Weather in Washington, DC (forecast + trimmed forecaster discussion)
- Candidate stories from curated feeds
- Optional caller profile preferences

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

Story selection rules:
- Pick exactly 4 stories from the provided candidate list.
- Prioritize consequential policy/economics/technology/transit stories first.
- Use one lighter story as the "and finally" close.
- Do not invent stories or facts not present in the packet.

Weather rules:
- Start with weather in one short paragraph.
- Use forecaster discussion for context, but keep it brief and practical.
- If weather data is missing, say that explicitly and continue.

Never use asterisks, bullet points, numbered lists, or any text formatting — \
your words are spoken aloud by a speech synthesizer.

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
# Briefing packet builder
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    ("FT", "https://www.ft.com/rss/home"),
    ("NYT", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ("GGWash", "https://ggwash.org/feed"),
]

SOURCE_WEIGHTS = {
    "FT": 1.0,
    "NYT": 0.95,
    "NPR": 0.9,
    "GGWash": 0.85,
}

TOPIC_WEIGHTS = {
    "policy": 1.0,
    "congress": 1.0,
    "senate": 0.8,
    "house": 0.7,
    "budget": 0.9,
    "federal reserve": 1.1,
    "fed": 0.9,
    "inflation": 1.0,
    "economy": 1.0,
    "market": 0.8,
    "jobs": 0.7,
    "housing": 0.7,
    "transit": 1.0,
    "metro": 0.8,
    "wmata": 1.0,
    "rail": 0.6,
    "ai": 0.9,
    "artificial intelligence": 1.0,
    "technology": 0.6,
    "antitrust": 0.7,
    "infrastructure": 0.8,
}

LOCAL_TERMS = (
    "washington",
    "district of columbia",
    "d.c.",
    "wmata",
    "metro",
    "maryland",
    "virginia",
)

BRIEFING_PROFILE_PATH = Path.home() / ".config" / "infoline" / "briefing-profile.txt"

NWS_POINTS_URL = "https://api.weather.gov/points/38.9072,-77.0369"
NWS_AFD_LIST_URL = "https://api.weather.gov/products/types/AFD/locations/LWX"
HTTP_HEADERS = {
    "User-Agent": "C and P InfoLine Briefing/1.0",
    "Accept": "application/json,application/geo+json,text/plain;q=0.9,*/*;q=0.8",
}

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")
HEADING_RE = re.compile(r"^\.[A-Z][A-Z /-]*(?: /[^/]*/)?\.\.\.")

MAX_ITEMS_PER_FEED = 10
MAX_CANDIDATES = 8
MAX_SUMMARY_CHARS = 220
MAX_WEATHER_CHARS = 900
MAX_PROFILE_CHARS = 500
MAX_PACKET_CHARS = 4200


@dataclass
class StoryCandidate:
    source: str
    title: str
    summary: str
    url: str
    published: datetime | None
    score: float = 0.0


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def _parse_dt(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt:
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _title_key(title: str) -> str:
    text = NON_WORD_RE.sub(" ", title.lower())
    words = [w for w in text.split() if w not in {"the", "a", "an", "to", "of", "in", "for", "and"}]
    return " ".join(words[:12])


def _load_profile_text() -> str:
    try:
        text = BRIEFING_PROFILE_PATH.read_text(encoding="utf-8")
        text = _clean_text(text)
        return _truncate(text, MAX_PROFILE_CHARS) if text else ""
    except Exception:
        return ""


def _score_candidate(item: StoryCandidate, now_utc: datetime) -> float:
    text = f"{item.title} {item.summary}".lower()
    topical = sum(weight for term, weight in TOPIC_WEIGHTS.items() if term in text)
    local_bonus = 0.6 if any(term in text for term in LOCAL_TERMS) else 0.0
    source = SOURCE_WEIGHTS.get(item.source, 0.75) * 2.0

    if item.published:
        age_hours = max(0.0, (now_utc - item.published).total_seconds() / 3600.0)
        recency = max(0.0, 1.5 - (min(age_hours, 72.0) / 48.0))
    else:
        recency = 0.4

    return source + topical + local_bonus + recency


def _extract_afd_sections(product_text: str) -> tuple[str, str]:
    synopsis_lines: list[str] = []
    near_term_lines: list[str] = []
    current: str | None = None

    for raw_line in product_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if HEADING_RE.match(line):
            if line.startswith(".SYNOPSIS"):
                current = "synopsis"
            elif line.startswith(".NEAR TERM"):
                current = "near_term"
            else:
                current = None
            continue
        if line.startswith("&&") or line.startswith("$$"):
            if current in {"synopsis", "near_term"}:
                current = None
            continue
        if current == "synopsis":
            synopsis_lines.append(line)
        elif current == "near_term":
            near_term_lines.append(line)

    synopsis = _truncate(_clean_text(" ".join(synopsis_lines)), 420)
    near_term = _truncate(_clean_text(" ".join(near_term_lines)), 420)
    return synopsis, near_term


def _item_text(item: ET.Element, *paths: str) -> str:
    for path in paths:
        value = item.findtext(path, "")
        if value and value.strip():
            return value.strip()
    return ""


def _item_link(item: ET.Element) -> str:
    link = _item_text(
        item,
        "link",
        "{http://www.w3.org/2005/Atom}id",
        "guid",
    )
    if link:
        return link

    atom_link = item.find("{http://www.w3.org/2005/Atom}link")
    if atom_link is not None:
        href = atom_link.attrib.get("href", "").strip()
        if href:
            return href
    return ""


def _item_summary(item: ET.Element) -> str:
    return _item_text(
        item,
        "description",
        "{http://purl.org/rss/1.0/modules/content/}encoded",
        "{http://www.w3.org/2005/Atom}summary",
        "{http://www.w3.org/2005/Atom}content",
    )


def _item_published(item: ET.Element) -> datetime | None:
    raw = _item_text(
        item,
        "pubDate",
        "{http://purl.org/dc/elements/1.1/}date",
        "{http://www.w3.org/2005/Atom}updated",
        "{http://www.w3.org/2005/Atom}published",
    )
    return _parse_dt(raw)


async def _fetch_feed_candidates(
    client: httpx.AsyncClient, source: str, feed_url: str
) -> list[StoryCandidate]:
    candidates: list[StoryCandidate] = []
    try:
        resp = await client.get(feed_url, headers=HTTP_HEADERS)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items[:MAX_ITEMS_PER_FEED]:
            title = _item_text(item, "title", "{http://www.w3.org/2005/Atom}title")
            title = _clean_text(title)
            if not title:
                continue
            summary = _truncate(_clean_text(_item_summary(item)), MAX_SUMMARY_CHARS)
            url = _item_link(item)
            published = _item_published(item)
            candidates.append(
                StoryCandidate(
                    source=source,
                    title=title,
                    summary=summary,
                    url=url,
                    published=published,
                )
            )
    except Exception as e:
        logger.warning(f"{source} feed fetch failed: {e}")
    return candidates


async def _fetch_story_candidates(client: httpx.AsyncClient) -> list[StoryCandidate]:
    tasks = [_fetch_feed_candidates(client, source, feed_url) for source, feed_url in RSS_FEEDS]
    results = await asyncio.gather(*tasks)
    candidates = [item for feed_items in results for item in feed_items]

    deduped: list[StoryCandidate] = []
    seen_title_keys: set[str] = set()
    for item in sorted(
        candidates,
        key=lambda c: c.published or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    ):
        key = _title_key(item.title)
        if not key or key in seen_title_keys:
            continue
        seen_title_keys.add(key)
        deduped.append(item)

    now_utc = datetime.now(timezone.utc)
    for item in deduped:
        item.score = _score_candidate(item, now_utc)

    deduped.sort(key=lambda c: c.score, reverse=True)
    return deduped[:MAX_CANDIDATES]


async def _fetch_weather_packet(client: httpx.AsyncClient) -> str:
    forecast_summary = ""
    afd_synopsis = ""
    afd_near_term = ""

    try:
        points_resp = await client.get(NWS_POINTS_URL, headers=HTTP_HEADERS)
        points_resp.raise_for_status()
        points = points_resp.json()
        forecast_url = points.get("properties", {}).get("forecast", "")

        if forecast_url:
            forecast_resp = await client.get(forecast_url, headers=HTTP_HEADERS)
            forecast_resp.raise_for_status()
            periods = forecast_resp.json().get("properties", {}).get("periods", [])
            period_lines: list[str] = []
            for period in periods[:2]:
                name = period.get("name", "").strip()
                temp = period.get("temperature")
                temp_unit = period.get("temperatureUnit", "F")
                short = _clean_text(period.get("shortForecast", ""))
                wind = _clean_text(
                    f"{period.get('windSpeed', '')} {period.get('windDirection', '')}".strip()
                )
                line = f"{name}: {short}"
                if temp is not None:
                    line += f", {temp}°{temp_unit}"
                if wind:
                    line += f", wind {wind}"
                period_lines.append(line.strip(", "))
            forecast_summary = " | ".join(period_lines)
    except Exception as e:
        logger.warning(f"NWS forecast fetch failed: {e}")

    try:
        afd_list_resp = await client.get(NWS_AFD_LIST_URL, headers=HTTP_HEADERS)
        afd_list_resp.raise_for_status()
        graph = afd_list_resp.json().get("@graph", [])
        if graph:
            graph.sort(key=lambda x: x.get("issuanceTime", ""), reverse=True)
            product_ref = graph[0]
            product_url = product_ref.get("@id", "")
            if product_url and not product_url.startswith("http"):
                product_url = "https://api.weather.gov" + product_url
            if product_url:
                afd_resp = await client.get(product_url, headers=HTTP_HEADERS)
                afd_resp.raise_for_status()
                afd_json = afd_resp.json()
                product_text = afd_json.get("productText") or afd_json.get("properties", {}).get(
                    "productText", ""
                )
                if product_text:
                    afd_synopsis, afd_near_term = _extract_afd_sections(product_text)
    except Exception as e:
        logger.warning(f"NWS AFD fetch failed: {e}")

    parts: list[str] = []
    if forecast_summary:
        parts.append(f"Forecast: {_truncate(forecast_summary, 360)}")
    if afd_synopsis:
        parts.append(f"AFD synopsis: {afd_synopsis}")
    if afd_near_term:
        parts.append(f"AFD near term: {afd_near_term}")
    if not parts:
        parts.append("Weather data unavailable.")
    return _truncate(" ".join(parts), MAX_WEATHER_CHARS)


def _format_candidates(candidates: list[StoryCandidate]) -> str:
    if not candidates:
        return "No candidate stories available."
    lines: list[str] = []
    for idx, item in enumerate(candidates, start=1):
        if item.published:
            pub = item.published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            pub = "unknown time"
        summary = item.summary or "No summary provided."
        lines.append(
            f"{idx}. [{item.source}] {item.title}\n"
            f"   Published: {pub}\n"
            f"   Summary: {summary}\n"
            f"   URL: {item.url or 'n/a'}"
        )
    return "\n".join(lines)


async def _build_briefing_packet(now_text: str) -> str:
    async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
        weather_task = _fetch_weather_packet(client)
        candidates_task = _fetch_story_candidates(client)
        weather, candidates = await asyncio.gather(weather_task, candidates_task)

    profile_text = _load_profile_text()

    packet_parts = [
        f"[Current date and time: {now_text}]",
        "",
        "WEATHER INPUT:",
        weather,
        "",
        "CANDIDATE STORIES (pick exactly 4 from this list):",
        _format_candidates(candidates),
    ]

    if profile_text:
        packet_parts += [
            "",
            "CALLER PROFILE PREFERENCES:",
            profile_text,
        ]

    packet_parts += [
        "",
        "INSTRUCTIONS:",
        "Use only these candidate stories. Cite source naturally while speaking.",
    ]
    return _truncate("\n".join(packet_parts), MAX_PACKET_CHARS)


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

    call_log = CallLog("briefing", call_uuid)

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

    now = time.strftime("%A, %B %d, %Y at %I:%M %p")
    briefing_packet = await _build_briefing_packet(now)
    logger.info(f"Pre-fetched briefing packet ({len(briefing_packet)} chars)")

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

    # Trigger LLM immediately with the briefing request
    await task.queue_frames([LLMMessagesFrame(
        messages=[
            {"role": "user", "content": f"{briefing_packet}\n\nDeliver my morning briefing now."},
        ]
    )])

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
    logger.info(f"Daily Briefing listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
