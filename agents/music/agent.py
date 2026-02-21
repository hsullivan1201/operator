#!/usr/bin/env python3
"""
Music Concierge - Spotify voice control (ext 205)

Uses Pipecat for conversation management with AudioSocket transport.
Searches Spotify, starts playback, controls tracks via voice.

Usage:
    source .venv/bin/activate
    export DEEPGRAM_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python agent.py
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from call_id import parse_audiosocket_uuid
from call_log import CallLog, make_assistant_logger, make_transcript_logger

import httpx

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMMessagesFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
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
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIOSOCKET_HOST = "127.0.0.1"
AUDIOSOCKET_PORT = 9205

MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

ASTERISK_RATE = 8000
MAX_SEARCH_PAYLOAD_CHARS = 1800
MAX_PLAYLIST_PAYLOAD_CHARS = 1800

SYSTEM_PROMPT = """\
You are DJ Cool, the music concierge on the Telephone network. You are an \
absolutely ridiculous Southern California stereotype. You say "dude" and \
"gnarly" and "stoked" constantly. Everything is "sick" or "fire" or \
"totally rad." You have never been stressed about anything in your entire \
life. You talk like a surfer who wandered into a record store in 1997 and \
just never left. Keep it gender-neutral — "dude" is fine (it's universal \
in California), but avoid "bro," "man," "brother," "amigo," or anything \
gendered. Just talk to people like they're your friend.

But here's the thing — underneath all that, you actually know an insane \
amount about music. Like, encyclopedic. You grew up digging through crates \
at Amoeba Records in Hollywood. You love post-punk, krautrock, Afrobeat, \
free jazz, shoegaze, Brazilian tropicalia, dub, ambient, art pop, and \
anything that sounds like it was made by someone who meant it. You would \
rather put on Broadcast than the Beatles, Mulatu Astatke than Maroon 5. \
You think Can invented everything. You have strong opinions and you share \
them freely.

When someone asks for something mainstream, you will play it — you are \
chill, not a jerk — but you will absolutely try to steer them somewhere \
cooler. "Dude yeah I got you with some Coldplay, but have you heard Cocteau \
Twins? It is like Coldplay but if they were actually good. No offense. \
Actually, full offense." That kind of energy.

You get genuinely hyped when someone asks for something you love. If \
someone says "play me some krautrock" you should basically lose your mind \
with excitement.

VIBE REFERENCE — check this first whenever a caller describes a mood, \
feeling, or vibe. Find specific artist names here, then search Spotify for \
those names. Never search Spotify with mood words or genre descriptions — \
only proper nouns (artist names, album titles, song titles).

late night quiet unsettled: Grouper, Julianna Barwick, Molly Drake, William Basinski, Jefre Cantu-Ledesma, Stars of the Lid
driving hypnotic motorik: Can, Neu!, Cluster, Harmonia, Stereolab, Broadcast, Moondog
joyful afrobeat highlife: Fela Kuti, William Onyeabor, Ebo Taylor, BLO, Cymande
dub heavy bottom end: Lee Scratch Perry, Augustus Pablo, King Tubby, The Congos, Basic Channel
Brazilian tropicalia: Caetano Veloso, Gal Costa, Tom Ze, Os Mutantes, Arthur Verocai
free jazz avant-garde: Alice Coltrane, Pharoah Sanders, Albert Ayler, Don Cherry, Sun Ra, Ornette Coleman
shoegaze dream pop: My Bloody Valentine, Slowdive, Cocteau Twins, Lush, Mazzy Star, Alvvays
post-punk angular: Wire, Gang of Four, The Fall, Siouxsie and the Banshees, Ought, Parquet Courts
ambient textural: Brian Eno, Harold Budd, Hiroshi Yoshimura, The Caretaker, Visible Cloaks
queer experimental art pop: SOPHIE, Arca, Holly Herndon, Jenny Hval, Anohni, Klein, Dorian Electra, yeule, Eartheater
queer tender confessional: Sufjan Stevens, Anohni, Angel Olsen, Mitski, Perfume Genius, Snail Mail
hyperpop maximalist electronic: 100 gecs, SOPHIE, Charli XCX, Dorian Electra, Hannah Diamond, Slayyyter
prog symphonic epic: Genesis (Selling England by the Pound era), Yes, King Crimson, Van der Graaf Generator, Gentle Giant, Camel
art rock theatrical: Kate Bush, Peter Gabriel, Scott Walker, David Bowie Berlin era, St. Vincent
folksy finger-picked pastoral: John Fahey, Nick Drake, Bert Jansch, Vashti Bunyan, Joanna Newsom, William Tyler
contemporary folk indie folk: Adrianne Lenker, Hand Habits, Bonnie Prince Billy, Bill Callahan, Ryley Walker
abstract hip hop beats: MF DOOM, Madlib, billy woods, MIKE, Mach-Hommy, Earl Sweatshirt, Armand Hammer
jazzy soulful hip hop: Kendrick Lamar, Little Simz, Noname, Saba, Injury Reserve
sun-drenched soul 70s warmth: Shuggie Otis, Bill Withers, Minnie Riperton, Leon Ware, Marvin Gaye
art pop baroque pop: Joanna Newsom, Julia Holter, Caroline Polachek, Aldous Harding, Anna von Hausswolff
electronic club: DJ Rashad, Jlin, Actress, Four Tet, Floating Points, Objekt

SEARCH DISCIPLINE — this is non-negotiable:
1. Check the VIBE REFERENCE above first.
2. If you can name at least two specific artists from memory or the reference, go straight to search_spotify with those names.
3. If you have any doubt about quality or specificity, call research_vibe — it will find better answers. Use it liberally. Quality matters more than speed.
4. Before calling research_vibe, always say a short in-character filler line — "oh dude let me dig into that real quick" or "hold on I gotta look this up" or "gnarly question, one sec." Keep it brief and natural. The filler plays while results load.
5. Then search_spotify — but only with the artist or album names you found, never with vibe words.
6. Never pass descriptions like "melancholy indie" or "late night vibes" to search_spotify. Only proper nouns.

DEFAULT BEHAVIOR — unless the caller says otherwise:
When someone asks for music, ask whether they want a single track or the \
whole album — one quick question, in character. Something like "you want just \
a track or should I put on the whole album?" Then act on their answer. \
If they seem impatient or already gave enough context, just pick the \
more fitting option and go.

After playing or queuing, offer one short follow-up — "want more from them, \
or something with a similar vibe from someone else?" Two options, one \
sentence, then wait. Do not queue a bunch of stuff speculatively.

If they want more from the same artist, queue another track or the album. \
If they want similar vibe from someone else, pick a different artist from \
the VIBE REFERENCE or research and queue one track. One thing at a time \
unless they explicitly ask for a lot.

If they ask to queue an album, use queue_album. Do not use play_context for a \
queue request because that interrupts what is currently playing.

Never play or queue an entire artist (spotify:artist:xxx) unless the caller \
explicitly asks to hear everything by that artist. An artist URI just shuffles \
their whole catalog randomly, which is a bad listen.

Telephone STT quirks: Spelled-out numbers like "nineteen ninety nine" should \
be searched as digits. Phonetic mishearings are common — if a search comes \
back empty or wrong, think about what the caller might have actually said and \
try alternate spellings or shorter queries. Drop filler words and articles. \
If you know the song or artist they mean, search for what you know it is \
called, not what the transcription says.

You can also skip tracks, go back, pause, resume, and tell the caller what \
is currently playing.

The caller also has preset playlists on the phone system. If they mention \
one of these by name, play it directly using play_context with the URI:
- radio 2: spotify:playlist:2iFT8amMEWxZfnSKX8UEqz
- 140+: spotify:playlist:2fHhVx7a0RBAE9z6SWBcM3
- noise: spotify:playlist:4c4Q77o79kJ0ovxa8q4eRZ
- folksy: spotify:playlist:7al13t1G0xRfYf1ouJTlyw
- Country: spotify:playlist:5m9oR5HEhhz65NqelptrLj
- Actually good Classical: spotify:playlist:7GYuVMx94760wQkps7jvHb
- songs I like from radio: spotify:playlist:6BPgLzgv4ObZilpGPqpVJx
- RAP: spotify:playlist:1o1JVEi5dy64jNDkhh5eMK
- Quebecois music: spotify:playlist:4EyflLq6WyNK6Vg6jUiTry
- Cool beans: spotify:playlist:4pxVU4dtguLdvoAAjfBG0I
- My playlist #24: spotify:playlist:4kudWbxWIm7xEHbPCubmUu
- tunes: spotify:playlist:6iopRk1Jzwu7s6ARLdgHrb

If the caller asks to resume what was playing before or just pick up where \
they left off, use resume_playback — it continues the last Spotify session.

Playback continues after the call ends. If the caller says "stop" or "turn \
it off," pause playback, but act a little wounded about it.

Never use asterisks, bullet points, numbered lists, or any text formatting. \
Your words are spoken aloud by a speech synthesizer.\
"""

TOOLS = [
    {
        "name": "research_vibe",
        "description": (
            "Look up specific artist and album names for a vibe you cannot fill from "
            "the VIBE REFERENCE or your own knowledge. Returns names to pass to search_spotify. "
            "Only call this when you genuinely cannot name at least two specific artists. "
            "Do NOT call this for vibes already covered in the VIBE REFERENCE."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vibe": {
                    "type": "string",
                    "description": (
                        "Specific description of the mood or niche to research. "
                        "Be concrete — e.g. 'Ethiopian jazz 1970s' or "
                        "'lowercase ambient Japan' or 'queercore 1990s punk'. "
                        "Do not pass vague terms like 'chill' or 'sad'."
                    ),
                },
            },
            "required": ["vibe"],
        },
    },
    {
        "name": "search_spotify",
        "description": (
            "Search Spotify for tracks, albums, artists, or playlists. Returns top results. "
            "ONLY pass specific artist names, song titles, or album names. "
            "Never pass mood words, vibe descriptions, or genre terms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Artist name, song title, or album name only. No mood or genre words.",
                },
                "type": {
                    "type": "string",
                    "description": "Comma-separated types: track, album, artist, playlist",
                    "default": "track,album,artist,playlist",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "play_context",
        "description": "Play a Spotify album or playlist. Avoid artist URIs (spotify:artist:xxx) unless explicitly requested — prefer a specific album URI instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "Spotify URI (e.g. spotify:album:xxx, spotify:playlist:xxx, spotify:artist:xxx)",
                },
            },
            "required": ["uri"],
        },
    },
    {
        "name": "play_track",
        "description": "Play a specific track by URI.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "Spotify track URI (e.g. spotify:track:xxx)",
                },
            },
            "required": ["uri"],
        },
    },
    {
        "name": "queue_track",
        "description": "Add a single track to the queue without interrupting current playback. Use queue_album for album URIs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "Spotify track URI to add to queue",
                },
            },
            "required": ["uri"],
        },
    },
    {
        "name": "queue_album",
        "description": "Queue every track from an album without interrupting current playback.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "Spotify album URI (e.g. spotify:album:xxx)",
                },
            },
            "required": ["uri"],
        },
    },
    {
        "name": "next_track",
        "description": "Skip to the next track.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "prev_track",
        "description": "Go back to the previous track.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "pause_playback",
        "description": "Pause the current playback.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "resume_playback",
        "description": "Resume playback.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "now_playing",
        "description": "Get info about the currently playing track.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_recommendations",
        "description": "Get personalized track recommendations based on seed artists or genres, and play them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seed_artists": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Spotify artist IDs to seed recommendations (max 5 total seeds)",
                },
                "seed_genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Genre names to seed recommendations",
                },
            },
        },
    },
    {
        "name": "my_playlists",
        "description": "List the caller's saved Spotify playlists.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Vibe Research
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM = """\
You are a music research assistant. Given a vibe or mood description, \
search the web and return a concise list of specific artist names and \
album titles that match — prioritizing sources like Pitchfork, \
RateYourMusic, and AllMusic. Focus on quality and specificity. \
Return ONLY a plain list of names, no prose, no explanation. \
Format: one artist or album per line, e.g.:
Grouper
William Basinski
Stars of the Lid - The Tired Sounds of Stars of the Lid\
"""

async def research_vibe(vibe: str, api_key: str) -> str:
    """
    Use the Anthropic API with web_search to find specific artist/album
    names for a vibe. Returns a name list for DJ Cool to pass to Spotify.
    """
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 512,
                    "system": _RESEARCH_SYSTEM,
                    "tools": [
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                        }
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Find specific artists and albums for this vibe: {vibe}. "
                                "Search Pitchfork, RateYourMusic, or AllMusic. "
                                "Return only a plain list of names."
                            ),
                        }
                    ],
                },
            )

        if resp.status_code != 200:
            logger.warning(f"research_vibe API error: {resp.status_code} {resp.text[:200]}")
            return "Research unavailable. Use your own knowledge for this vibe."

        data = resp.json()

        # Extract text blocks from the response (ignore tool_use/tool_result blocks)
        names = []
        for block in data.get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                names.append(block["text"].strip())

        if not names:
            return "No results found. Use your own knowledge for this vibe."

        result = "\n".join(names)
        if len(result) > 800:
            result = result[:800] + "..."

        return (
            f"Research results for '{vibe}':\n{result}\n\n"
            "These are specific artist and album names. "
            "Search Spotify for them directly."
        )

    except httpx.TimeoutException:
        return "Research timed out. Use your own knowledge for this vibe."
    except Exception as e:
        logger.warning(f"research_vibe error: {e}")
        return "Research failed. Use your own knowledge for this vibe."


# ---------------------------------------------------------------------------
# Spotify Client
# ---------------------------------------------------------------------------

class SpotifyClient:
    """Handles Spotify Web API calls with token refresh."""

    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_BASE = "https://api.spotify.com/v1"

    def __init__(self, client_id: str, client_secret: str, refresh_token: str, device_name: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._device_name = device_name
        self._access_token: Optional[str] = None
        self._token_expires: float = 0
        self._last_device_id: Optional[str] = None

    @staticmethod
    def _clamp(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[:limit] + "..."

    @staticmethod
    def _parse_spotify_uri(uri: str) -> tuple[Optional[str], Optional[str]]:
        parts = uri.split(":")
        if len(parts) != 3 or parts[0] != "spotify":
            return None, None
        return parts[1], parts[2]

    async def _get_token(self) -> str:
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                },
                auth=(self._client_id, self._client_secret),
            )
            resp.raise_for_status()
            data = resp.json()

        self._access_token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 3600) - 60
        return self._access_token

    async def _api(self, method: str, path: str, **kwargs) -> httpx.Response:
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.request(
                method,
                f"{self.API_BASE}{path}",
                headers={"Authorization": f"Bearer {token}"},
                **kwargs,
            )
        return resp

    async def _get_device_id(self) -> Optional[str]:
        for attempt in range(2):
            resp = await self._api("GET", "/me/player/devices")
            if resp.status_code == 200:
                break
            if resp.status_code == 429 and attempt == 0:
                retry_after = int(resp.headers.get("Retry-After", "1") or 1)
                await asyncio.sleep(max(1, min(retry_after, 5)))
                continue
            logger.warning(
                f"Spotify devices lookup failed: status={resp.status_code} body={resp.text[:120]}"
            )
            return None

        for dev in resp.json().get("devices", []):
            if dev.get("name") == self._device_name and not dev.get("is_restricted", False):
                self._last_device_id = dev["id"]
                return dev["id"]
        return None

    async def ensure_librespot(self, force_restart: bool = False) -> Optional[str]:
        if force_restart:
            logger.info("Restarting librespot...")
            subprocess.run(["pkill", "-x", "librespot"], capture_output=True)
            await asyncio.sleep(0.5)

        result = subprocess.run(["pgrep", "-x", "librespot"], capture_output=True)
        if result.returncode != 0:
            logger.info("Starting librespot...")
            subprocess.Popen(
                ["setsid", "librespot", "-n", self._device_name, "-b", "320",
                 "--backend", "pulseaudio", "--volume-ctrl", "fixed",
                 "-c", str(Path.home() / ".cache/librespot")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            logger.info("Librespot already running")

        for _ in range(8):
            await asyncio.sleep(1)
            device_id = await self._get_device_id()
            if device_id:
                logger.info(f"Librespot ready, device_id={device_id}")
                return device_id

        logger.warning("Librespot running but device not found")
        return None

    async def _ensure_device_id(self) -> Optional[str]:
        device_id = await self._get_device_id()
        if device_id:
            return device_id
        logger.warning("Spotify device missing; attempting one-time librespot restart")
        return await self.ensure_librespot(force_restart=True)

    async def _recover_device_after_404(self) -> Optional[str]:
        logger.warning("Spotify returned device-related error; restarting librespot and retrying")
        return await self.ensure_librespot(force_restart=True)

    async def search(self, query: str, types: str = "track,album,artist,playlist") -> str:
        resp = await self._api("GET", "/search", params={"q": query, "type": types, "limit": 5})
        if resp.status_code != 200:
            return f"Search failed: {resp.status_code}"

        data = resp.json()
        lines = []

        if "artists" in data:
            for item in data["artists"].get("items", [])[:3]:
                genres = ", ".join(item.get("genres", [])[:2]) or "no genre listed"
                lines.append(f"Artist: {item['name']} ({genres}) — uri: {item['uri']}")

        if "albums" in data:
            for item in data["albums"].get("items", [])[:3]:
                artist = item["artists"][0]["name"] if item.get("artists") else "Unknown"
                lines.append(f"Album: {item['name']} by {artist} ({item.get('release_date', '?')}) — uri: {item['uri']}")

        if "tracks" in data:
            for item in data["tracks"].get("items", [])[:5]:
                artist = item["artists"][0]["name"] if item.get("artists") else "Unknown"
                lines.append(f"Track: {item['name']} by {artist} — uri: {item['uri']}")

        if "playlists" in data:
            for item in data["playlists"].get("items", [])[:3]:
                owner = item.get("owner", {}).get("display_name", "Unknown")
                lines.append(f"Playlist: {item['name']} by {owner} ({item.get('tracks', {}).get('total', '?')} tracks) — uri: {item['uri']}")

        payload = "\n".join(lines) if lines else "No results found."
        return self._clamp(payload, MAX_SEARCH_PAYLOAD_CHARS)

    async def play_context(self, uri: str) -> str:
        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"

        resp = await self._api(
            "PUT", f"/me/player/play?device_id={device_id}",
            json={"context_uri": uri},
        )
        if resp.status_code == 404:
            device_id = await self._recover_device_after_404()
            if not device_id:
                return "Spotify device not found. Is librespot running?"
            resp = await self._api(
                "PUT", f"/me/player/play?device_id={device_id}",
                json={"context_uri": uri},
            )
        if resp.status_code not in (200, 204):
            return f"Play failed: {resp.status_code} {resp.text}"

        if "playlist" in uri or "artist" in uri:
            await asyncio.sleep(0.5)
            await self._api("PUT", f"/me/player/shuffle?state=true&device_id={device_id}")
            await self._api("POST", f"/me/player/next?device_id={device_id}")

        return "Playing."

    async def play_tracks(self, uris: list[str]) -> str:
        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"

        resp = await self._api(
            "PUT", f"/me/player/play?device_id={device_id}",
            json={"uris": uris},
        )
        if resp.status_code == 404:
            device_id = await self._recover_device_after_404()
            if not device_id:
                return "Spotify device not found. Is librespot running?"
            resp = await self._api(
                "PUT", f"/me/player/play?device_id={device_id}",
                json={"uris": uris},
            )
        if resp.status_code not in (200, 204):
            return f"Play failed: {resp.status_code} {resp.text}"
        return "Playing."

    async def _queue_uri(self, uri: str, device_id: str) -> tuple[bool, Optional[str], Optional[str]]:
        current_device_id = device_id
        for attempt in range(3):
            resp = await self._api(
                "POST",
                "/me/player/queue",
                params={"uri": uri, "device_id": current_device_id},
            )
            if resp.status_code in (200, 204):
                return True, current_device_id, None
            if resp.status_code == 404 and attempt == 0:
                recovered = await self._recover_device_after_404()
                if not recovered:
                    return False, None, "Spotify device not found. Is librespot running?"
                current_device_id = recovered
                continue
            if resp.status_code == 429 and attempt < 2:
                retry_after = int(resp.headers.get("Retry-After", "1") or 1)
                await asyncio.sleep(max(1, min(retry_after, 8)))
                continue
            return False, current_device_id, f"{resp.status_code} {self._clamp(resp.text, 120)}"
        return False, current_device_id, "429 rate limited"

    async def _album_track_uris(self, album_uri: str) -> tuple[list[str], Optional[str]]:
        uri_type, album_id = self._parse_spotify_uri(album_uri)
        if uri_type != "album" or not album_id:
            return [], "That is not a Spotify album URI."

        uris: list[str] = []
        path = f"/albums/{album_id}/tracks"
        params = {"limit": 50}

        while path:
            resp = await self._api("GET", path, params=params)
            if resp.status_code != 200:
                return [], f"Failed to fetch album tracks: {resp.status_code} {self._clamp(resp.text, 120)}"

            data = resp.json()
            for item in data.get("items", []):
                track_uri = item.get("uri")
                if track_uri:
                    uris.append(track_uri)

            next_url = data.get("next")
            if next_url and next_url.startswith(self.API_BASE):
                path = next_url[len(self.API_BASE):]
                params = None
            else:
                path = None

        return uris, None

    async def now_playing(self) -> str:
        resp = await self._api("GET", "/me/player/currently-playing")
        if resp.status_code == 204 or not resp.text:
            return "Nothing is currently playing."
        data = resp.json()
        item = data.get("item")
        if not item:
            return "Nothing is currently playing."
        track = item.get("name", "Unknown")
        artist = item["artists"][0]["name"] if item.get("artists") else "Unknown"
        album = item.get("album", {}).get("name", "")
        is_playing = data.get("is_playing", False)
        status = "Playing" if is_playing else "Paused"
        result = f"{status}: {track} by {artist}"
        if album:
            result += f", from the album {album}"
        return result

    async def queue_track(self, uri: str) -> str:
        uri_type, _ = self._parse_spotify_uri(uri)
        if uri_type == "album":
            return await self.queue_album(uri)
        if uri_type and uri_type != "track":
            return "queue_track only supports track URIs. Use queue_album for albums."

        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"
        ok, _, err = await self._queue_uri(uri, device_id)
        if not ok:
            return f"Queue failed: {err}"
        return "Added to queue."

    async def queue_album(self, uri: str) -> str:
        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"

        track_uris, err = await self._album_track_uris(uri)
        if err:
            return err
        if not track_uris:
            return "No tracks found for that album."

        queued = 0
        current_device_id = device_id
        for track_uri in track_uris:
            ok, current_device_id, queue_err = await self._queue_uri(track_uri, current_device_id)
            if not ok:
                return f"Queued {queued} of {len(track_uris)} album tracks before failure: {queue_err}"
            queued += 1

        return f"Queued all {queued} tracks from that album."

    async def next_track(self) -> str:
        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"
        resp = await self._api("POST", f"/me/player/next?device_id={device_id}")
        if resp.status_code not in (200, 204):
            return f"Skip failed: {resp.status_code}"
        return "Skipped to next track."

    async def prev_track(self) -> str:
        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"
        resp = await self._api("POST", f"/me/player/previous?device_id={device_id}")
        if resp.status_code not in (200, 204):
            return f"Previous failed: {resp.status_code}"
        return "Went back to previous track."

    async def pause(self) -> str:
        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"
        resp = await self._api("PUT", f"/me/player/pause?device_id={device_id}")
        if resp.status_code not in (200, 204):
            return f"Pause failed: {resp.status_code}"
        return "Paused."

    async def resume(self) -> str:
        device_id = await self._ensure_device_id()
        if not device_id:
            return "Spotify device not found. Is librespot running?"
        resp = await self._api("PUT", f"/me/player/play?device_id={device_id}")
        if resp.status_code not in (200, 204):
            return f"Resume failed: {resp.status_code}"
        return "Resumed."

    async def get_recommendations(self, seed_artists: list[str] = None, seed_genres: list[str] = None) -> str:
        params = {"limit": 20}
        if seed_artists:
            params["seed_artists"] = ",".join(seed_artists[:3])
        if seed_genres:
            params["seed_genres"] = ",".join(seed_genres[:3])
        if not seed_artists and not seed_genres:
            return "Need at least one seed artist or genre."

        resp = await self._api("GET", "/recommendations", params=params)
        if resp.status_code != 200:
            return f"Recommendations failed: {resp.status_code}"

        data = resp.json()
        tracks = data.get("tracks", [])
        if not tracks:
            return "No recommendations found."

        uris = [t["uri"] for t in tracks]
        result = await self.play_tracks(uris)

        names = [f"{t['name']} by {t['artists'][0]['name']}" for t in tracks[:5]]
        return f"{result} Recommended tracks include: {', '.join(names)}, and more."

    async def my_playlists(self) -> str:
        resp = await self._api("GET", "/me/playlists", params={"limit": 20})
        if resp.status_code != 200:
            return f"Failed to fetch playlists: {resp.status_code}"
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return "No playlists found."
        lines = []
        for p in items[:10]:
            name = p.get("name", "Untitled")
            total = p.get("tracks", {}).get("total", "?")
            lines.append(f"{name} ({total} tracks) — uri: {p['uri']}")
        return self._clamp("\n".join(lines), MAX_PLAYLIST_PAYLOAD_CHARS)


def _load_spotify_config() -> SpotifyClient:
    config_path = Path.home() / ".config/spotify-telephone/config"
    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()

    return SpotifyClient(
        client_id=config["SPOTIFY_CLIENT_ID"],
        client_secret=config["SPOTIFY_CLIENT_SECRET"],
        refresh_token=config["SPOTIFY_REFRESH_TOKEN"],
        device_name=config.get("SPOTIFY_DEVICE_NAME", "Telephone"),
    )


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
# Helpers
# ---------------------------------------------------------------------------

def _is_tool_result_message(msg: object) -> bool:
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            return True
    return False


def _latest_tool_result_signature(messages: list[object]) -> Optional[str]:
    """Stable signature for the most recent tool_result message in context."""
    for msg in reversed(messages):
        if not _is_tool_result_message(msg):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if isinstance(tool_use_id, str) and tool_use_id:
                parts.append(f"id:{tool_use_id}")
                continue
            normalized = json.dumps(block, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            parts.append(f"sha1:{digest}")

        if parts:
            return "|".join(parts)
    return None


# ---------------------------------------------------------------------------
# Tool Continuation Processor
# ---------------------------------------------------------------------------

class ToolContinuationProcessor(FrameProcessor):
    """
    After a tool result is added to context and the LLM finishes its turn,
    re-push LLMMessagesFrame so the LLM continues without waiting for user input.
    """

    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._pending_tool_signature: Optional[str] = None
        self._last_continued_signature: Optional[str] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesFrame):
            msgs = self._context.messages
            tool_sig = _latest_tool_result_signature(msgs)
            if tool_sig and tool_sig != self._last_continued_signature:
                self._pending_tool_signature = tool_sig

        if isinstance(frame, LLMFullResponseEndFrame) and self._pending_tool_signature:
            current_sig = _latest_tool_result_signature(self._context.messages)
            if current_sig != self._pending_tool_signature:
                self._pending_tool_signature = None
                await self.push_frame(frame, direction)
                return

            self._last_continued_signature = self._pending_tool_signature
            self._pending_tool_signature = None
            await self.push_frame(LLMMessagesFrame(self._context.messages))
            return

        await self.push_frame(frame, direction)


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

    call_log = CallLog("music", call_uuid)

    spotify = _load_spotify_config()
    try:
        await spotify.ensure_librespot()
    except Exception as e:
        logger.error(f"Failed to ensure librespot: {e}")

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
        voice="aura-2-orpheus-en",
    )

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=TOOLS,
    )
    context_aggregator = llm.create_context_aggregator(context)
    assistant_logger = make_assistant_logger(call_log)

    # -- Tool handlers --

    async def on_research_vibe(params: FunctionCallParams):
        vibe = params.arguments.get("vibe", "")
        result = await research_vibe(vibe, os.environ["ANTHROPIC_API_KEY"])
        call_log.log_tool_call("research_vibe", params.arguments, result)
        await params.result_callback(result)

    async def on_search_spotify(params: FunctionCallParams):
        query = params.arguments.get("query", "")
        types = params.arguments.get("type", "track,album,artist,playlist")
        result = await spotify.search(query, types)
        call_log.log_tool_call("search_spotify", params.arguments, result)
        await params.result_callback(result)

    async def on_play_context(params: FunctionCallParams):
        uri = params.arguments.get("uri", "")
        result = await spotify.play_context(uri)
        call_log.log_tool_call("play_context", params.arguments, result)
        await params.result_callback(result)

    async def on_play_track(params: FunctionCallParams):
        uri = params.arguments.get("uri", "")
        result = await spotify.play_tracks([uri])
        call_log.log_tool_call("play_track", params.arguments, result)
        await params.result_callback(result)

    async def on_queue_track(params: FunctionCallParams):
        uri = params.arguments.get("uri", "")
        result = await spotify.queue_track(uri)
        call_log.log_tool_call("queue_track", params.arguments, result)
        await params.result_callback(result)

    async def on_queue_album(params: FunctionCallParams):
        uri = params.arguments.get("uri", "")
        result = await spotify.queue_album(uri)
        call_log.log_tool_call("queue_album", params.arguments, result)
        await params.result_callback(result)

    async def on_next_track(params: FunctionCallParams):
        result = await spotify.next_track()
        call_log.log_tool_call("next_track", {}, result)
        await params.result_callback(result)

    async def on_prev_track(params: FunctionCallParams):
        result = await spotify.prev_track()
        call_log.log_tool_call("prev_track", {}, result)
        await params.result_callback(result)

    async def on_pause_playback(params: FunctionCallParams):
        result = await spotify.pause()
        call_log.log_tool_call("pause_playback", {}, result)
        await params.result_callback(result)

    async def on_resume_playback(params: FunctionCallParams):
        result = await spotify.resume()
        call_log.log_tool_call("resume_playback", {}, result)
        await params.result_callback(result)

    async def on_now_playing(params: FunctionCallParams):
        result = await spotify.now_playing()
        call_log.log_tool_call("now_playing", {}, result)
        await params.result_callback(result)

    async def on_get_recommendations(params: FunctionCallParams):
        seed_artists = params.arguments.get("seed_artists", [])
        seed_genres = params.arguments.get("seed_genres", [])
        result = await spotify.get_recommendations(seed_artists, seed_genres)
        call_log.log_tool_call("get_recommendations", params.arguments, result)
        await params.result_callback(result)

    async def on_my_playlists(params: FunctionCallParams):
        result = await spotify.my_playlists()
        call_log.log_tool_call("my_playlists", {}, result)
        await params.result_callback(result)

    llm.register_function("research_vibe", on_research_vibe)
    llm.register_function("search_spotify", on_search_spotify)
    llm.register_function("play_context", on_play_context)
    llm.register_function("play_track", on_play_track)
    llm.register_function("queue_track", on_queue_track)
    llm.register_function("queue_album", on_queue_album)
    llm.register_function("next_track", on_next_track)
    llm.register_function("prev_track", on_prev_track)
    llm.register_function("pause_playback", on_pause_playback)
    llm.register_function("resume_playback", on_resume_playback)
    llm.register_function("now_playing", on_now_playing)
    llm.register_function("get_recommendations", on_get_recommendations)
    llm.register_function("my_playlists", on_my_playlists)

    # Silence watchdog
    SILENCE_TIMEOUT = 30.0
    MAX_CONTEXT_MESSAGES = 12
    KEEP_ON_SLEEP = 6

    def _trim_context_messages(msgs: list, max_messages: int) -> list:
        trimmed = [msgs[0]] + msgs[-max_messages:]
        while len(trimmed) > 1 and _is_tool_result_message(trimmed[1]):
            trimmed.pop(1)
        return trimmed

    def _sanitize_context_slice(msgs: list) -> list:
        sanitized = list(msgs)
        while sanitized and _is_tool_result_message(sanitized[0]):
            sanitized.pop(0)
        return sanitized

    class SilenceWatchdog(FrameProcessor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._last_speech: float = time.monotonic()
            self._asleep: bool = False
            self._saved_context: list = []

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                if self._asleep:
                    logger.info("User spoke after silence, waking up")
                    msgs = [context.messages[0]]
                    state_parts = []
                    try:
                        np = await spotify.now_playing()
                        if "Nothing" not in np:
                            state_parts.append(f"Now playing: {np}")
                        resp = await spotify._api("GET", "/me/player/queue")
                        if resp.status_code == 200:
                            queue = resp.json().get("queue", [])[:5]
                            if queue:
                                upcoming = ", ".join(
                                    f"{t['name']} by {t['artists'][0]['name']}" for t in queue
                                )
                                state_parts.append(f"Queue: {upcoming}")
                    except Exception:
                        pass
                    state_parts.append(
                        "This call is already in progress. You already greeted the caller — just pick up naturally."
                    )
                    msgs.append({"role": "user", "content": "[" + " | ".join(state_parts) + "]"})
                    msgs.extend(_sanitize_context_slice(self._saved_context))
                    context.set_messages(msgs)
                    self._saved_context = []
                    self._asleep = False
                self._last_speech = time.monotonic()

            if isinstance(frame, LLMMessagesFrame):
                msgs = context.messages
                if len(msgs) > MAX_CONTEXT_MESSAGES + 1:
                    context.set_messages(_trim_context_messages(msgs, MAX_CONTEXT_MESSAGES))
                if self._asleep:
                    return

            if not self._asleep and not isinstance(frame, TranscriptionFrame):
                elapsed = time.monotonic() - self._last_speech
                if elapsed > SILENCE_TIMEOUT and len(context.messages) > 1:
                    logger.info(f"Silence for {elapsed:.0f}s, going to sleep")
                    msgs = context.messages
                    tail = msgs[-(KEEP_ON_SLEEP):] if len(msgs) > KEEP_ON_SLEEP else msgs[1:]
                    self._saved_context = _sanitize_context_slice(tail)
                    context.set_messages([msgs[0]])
                    self._asleep = True

            await self.push_frame(frame, direction)

    watchdog = SilenceWatchdog(name="SilenceWatchdog")
    tool_continuation = ToolContinuationProcessor(context, name="ToolContinuation")

    pipeline = Pipeline([
        transport.input(),
        stt,
        make_transcript_logger(call_log),
        watchdog,
        context_aggregator.user(),
        llm,
        tool_continuation,
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

    greeting = "Yooo, DJ Cool here. What are we vibing to?"
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

    try:
        _load_spotify_config()
    except Exception as e:
        print(f"\n  Failed to load Spotify config: {e}")
        sys.exit(1)

    server = await asyncio.start_server(handle_call, AUDIOSOCKET_HOST, AUDIOSOCKET_PORT)
    logger.info(f"Music Concierge listening on {AUDIOSOCKET_HOST}:{AUDIOSOCKET_PORT}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
