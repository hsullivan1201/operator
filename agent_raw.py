#!/usr/bin/env python3
"""
Operator - Voice AI agent for Asterisk via AudioSocket

Listens on a TCP port for AudioSocket connections from Asterisk.
Streams caller audio to Deepgram for STT, sends transcripts to Claude,
and plays back TTS audio through the call.

Usage:
    source .venv/bin/activate
    export DEEPGRAM_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export DEEPINFRA_API_KEY=...  # optional, for Kokoro TTS
    python operator.py
"""

import asyncio
import struct
import json
import logging
import os
import sys
import io
import wave
import time

import numpy as np
import httpx
import websockets
from anthropic import AsyncAnthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUDIOSOCKET_HOST = "127.0.0.1"
AUDIOSOCKET_PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_AUDIO = 0x10
MSG_ERROR = 0x11

# Audio
ASTERISK_RATE = 8000  # slin16 from Asterisk
SAMPLE_WIDTH = 2  # 16-bit

# Deepgram
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

# Kokoro TTS via DeepInfra (OpenAI-compatible endpoint)
KOKORO_URL = "https://api.deepinfra.com/v1/openai/audio/speech"
KOKORO_SAMPLE_RATE = 24000

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
                    "description": "The extension number to transfer to (e.g. '7' for KEXP radio)",
                }
            },
            "required": ["extension"],
        },
    }
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("operator")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample(audio: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample 16-bit PCM using numpy linear interpolation."""
    if src_rate == dst_rate:
        return audio
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float64)
    duration = len(samples) / src_rate
    new_len = int(duration * dst_rate)
    if new_len == 0:
        return b""
    indices = np.linspace(0, len(samples) - 1, new_len)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw PCM and sample rate from a WAV file."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as w:
            rate = w.getframerate()
            pcm = w.readframes(w.getnframes())
            return pcm, rate


# ---------------------------------------------------------------------------
# TTS backends
# ---------------------------------------------------------------------------

async def kokoro_tts(text: str) -> bytes:
    """Kokoro TTS via DeepInfra → returns 8 kHz slin16."""
    api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPINFRA_API_KEY not set")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            KOKORO_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "hexgrad/Kokoro-82M",
                "input": text,
                "voice": "af_heart",
                "response_format": "pcm",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        return resample(resp.content, KOKORO_SAMPLE_RATE, ASTERISK_RATE)


PIPER_MODEL = os.path.join(os.path.dirname(__file__), "voices", "en_US-lessac-medium.onnx")


async def piper_tts(text: str) -> bytes:
    """Piper TTS (local, good quality) → returns 8 kHz slin16."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "piper",
        "--model", PIPER_MODEL,
        "--output_file", "/dev/stdout",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(text.encode())
    if proc.returncode != 0:
        raise RuntimeError(f"piper failed: {stderr.decode()}")
    pcm, rate = wav_to_pcm(stdout)
    return resample(pcm, rate, ASTERISK_RATE)


async def espeak_tts(text: str) -> bytes:
    """espeak-ng TTS (local fallback) → returns 8 kHz slin16."""
    proc = await asyncio.create_subprocess_exec(
        "espeak-ng", "--stdout", "-v", "en-us", "-s", "150", text,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {stderr.decode()}")
    pcm, rate = wav_to_pcm(stdout)
    return resample(pcm, rate, ASTERISK_RATE)


async def text_to_speech(text: str) -> bytes:
    """Try Kokoro, then Piper, fall back to espeak-ng."""
    for backend, fn in [("kokoro", kokoro_tts), ("piper", piper_tts), ("espeak-ng", espeak_tts)]:
        try:
            audio = await fn(text)
            logger.info(f"TTS [{backend}]: {len(audio)} bytes")
            return audio
        except Exception as e:
            logger.warning(f"TTS [{backend}] failed: {e}")
    logger.error("All TTS backends failed")
    return b""


# ---------------------------------------------------------------------------
# Call handler
# ---------------------------------------------------------------------------

class OperatorCall:
    """Handles one phone call through the full voice pipeline."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.call_uuid: str | None = None
        self.running = True
        self.is_speaking = False

        # Conversation state
        self.conversation: list[dict] = []
        self.anthropic = AsyncAnthropic()

        # STT state
        self.deepgram_ws = None
        self.pending_transcript = ""
        self.transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._debounce_handle: asyncio.Task | None = None

    # -- Main entry point ---------------------------------------------------

    async def run(self):
        logger.info("New call connected")

        # First message should be the UUID
        msg_type, payload = await self._read_message()
        if msg_type == MSG_UUID:
            self.call_uuid = payload.decode("utf-8", errors="replace").strip("\x00")
            logger.info(f"Call UUID: {self.call_uuid}")

        # Connect Deepgram
        await self._connect_deepgram()

        # Run pipeline tasks
        tasks = [
            asyncio.create_task(self._read_audio_loop(), name="audio-in"),
            asyncio.create_task(self._read_transcripts(), name="stt"),
            asyncio.create_task(self._conversation_loop(), name="llm"),
        ]

        # Greeting
        await self._speak("Operator. How may I help you?")

        # Wait until something ends (hangup, error, etc.)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        self.running = False
        for t in pending:
            t.cancel()
        for t in done:
            if not t.cancelled() and t.exception():
                logger.error(f"Task {t.get_name()} error: {t.exception()}")

        # Cleanup Deepgram
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
            except Exception:
                pass

    # -- AudioSocket I/O ----------------------------------------------------

    async def _read_message(self) -> tuple[int, bytes]:
        header = await self.reader.readexactly(3)
        msg_type = header[0]
        msg_len = struct.unpack(">H", header[1:3])[0]
        payload = await self.reader.readexactly(msg_len) if msg_len > 0 else b""
        return msg_type, payload

    async def _read_audio_loop(self):
        """Continuously read audio from Asterisk and forward to Deepgram."""
        try:
            while self.running:
                msg_type, payload = await self._read_message()

                if msg_type == MSG_HANGUP:
                    logger.info("Caller hung up")
                    self.running = False
                    return
                elif msg_type == MSG_AUDIO:
                    if not self.is_speaking and self.deepgram_ws:
                        try:
                            await self.deepgram_ws.send(payload)
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Deepgram connection lost")
                elif msg_type == MSG_ERROR:
                    logger.error("AudioSocket error from Asterisk")
                    self.running = False
                    return
        except asyncio.IncompleteReadError:
            logger.info("AudioSocket connection closed")
            self.running = False

    async def _send_audio(self, audio: bytes):
        """Send 8 kHz slin16 audio back through AudioSocket at real-time pace."""
        chunk_samples = 160  # 20 ms at 8 kHz
        chunk_bytes = chunk_samples * SAMPLE_WIDTH
        start = time.monotonic()
        sent = 0

        for i in range(0, len(audio), chunk_bytes):
            if not self.running:
                break
            chunk = audio[i : i + chunk_bytes]
            header = struct.pack(">BH", MSG_AUDIO, len(chunk))
            self.writer.write(header + chunk)
            await self.writer.drain()

            sent += len(chunk) // SAMPLE_WIDTH
            target = sent / ASTERISK_RATE
            elapsed = time.monotonic() - start
            if target > elapsed:
                await asyncio.sleep(target - elapsed)

    # -- Deepgram STT -------------------------------------------------------

    async def _connect_deepgram(self):
        api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set")

        params = (
            f"encoding=linear16&sample_rate={ASTERISK_RATE}&channels=1"
            f"&model=nova-2&smart_format=true"
            f"&endpointing=200&interim_results=true"
        )
        self.deepgram_ws = await websockets.connect(
            f"{DEEPGRAM_URL}?{params}",
            additional_headers={"Authorization": f"Token {api_key}"},
        )
        logger.info("Deepgram connected")

    async def _read_transcripts(self):
        """Read Deepgram messages, accumulate finals, debounce into utterances."""
        try:
            async for raw in self.deepgram_ws:
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "Results":
                    alt = data["channel"]["alternatives"][0]
                    text = alt.get("transcript", "")
                    is_final = data.get("is_final", False)

                    if text and is_final:
                        self.pending_transcript += " " + text
                        logger.info(f"[STT final] {text}")
                        # Reset debounce timer
                        if self._debounce_handle:
                            self._debounce_handle.cancel()
                        self._debounce_handle = asyncio.create_task(
                            self._debounce_flush()
                        )

                elif msg_type == "UtteranceEnd":
                    # Deepgram detected end of utterance — flush immediately
                    if self._debounce_handle:
                        self._debounce_handle.cancel()
                    await self._flush_transcript()

        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram connection closed")

    async def _debounce_flush(self):
        """Flush transcript after a short silence."""
        await asyncio.sleep(0.8)
        await self._flush_transcript()

    async def _flush_transcript(self):
        """Push accumulated transcript to the conversation queue."""
        text = self.pending_transcript.strip()
        self.pending_transcript = ""
        if text:
            logger.info(f"[utterance] {text}")
            await self.transcript_queue.put(text)

    # -- LLM + TTS ----------------------------------------------------------

    async def _conversation_loop(self):
        """Wait for transcripts, get Claude responses, speak them."""
        while self.running:
            try:
                text = await asyncio.wait_for(self.transcript_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self.deepgram_ws:
                    try:
                        await self.deepgram_ws.send(json.dumps({"type": "KeepAlive"}))
                    except Exception:
                        pass
                continue

            logger.info(f"[user] {text}")
            self.conversation.append({"role": "user", "content": text})

            try:
                await self._get_response_and_act()
            except Exception as e:
                logger.error(f"Claude error: {e}", exc_info=True)
                await self._speak("I'm sorry, I'm having trouble. Could you try again?")

    async def _get_response_and_act(self):
        """Call Claude, speak text, handle tool use."""
        now = time.strftime("%A, %B %d, %Y at %I:%M %p")
        response = await self.anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SYSTEM_PROMPT.format(now=now),
            messages=self.conversation,
            tools=TOOLS,
        )

        # Process response blocks — there may be text + tool_use
        assistant_content = response.content
        self.conversation.append({"role": "assistant", "content": assistant_content})

        # Speak any text blocks first
        for block in assistant_content:
            if block.type == "text" and block.text:
                logger.info(f"[claude] {block.text}")
                await self._speak(block.text)

        # Handle tool calls
        for block in assistant_content:
            if block.type == "tool_use":
                result = await self._handle_tool(block.name, block.input)
                logger.info(f"[tool] {block.name}({block.input}) → {result}")
                # Send tool result back to Claude
                self.conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    ],
                })
                # If transfer succeeded, we're done
                if block.name == "transfer_call" and "transferred" in result.lower():
                    return

    async def _handle_tool(self, name: str, args: dict) -> str:
        """Execute a tool call and return the result string."""
        if name == "transfer_call":
            return await self._transfer_call(args.get("extension", ""))
        return f"Unknown tool: {name}"

    async def _transfer_call(self, extension: str) -> str:
        """Redirect the Asterisk channel to another extension."""
        valid = {"1", "2", "4", "5", "6", "7", "8"}
        if extension not in valid:
            return f"Invalid extension {extension}. Valid: {', '.join(sorted(valid))}"

        # Find our active channel
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
            return "Could not find active channel to transfer."

        logger.info(f"Redirecting {channel_name} → extension {extension}")
        proc = await asyncio.create_subprocess_exec(
            "sudo", "asterisk", "-rx",
            f"channel redirect {channel_name} internal,{extension},1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            self.running = False
            return f"Successfully transferred to extension {extension}."
        return f"Transfer failed: {stderr.decode()}"

    async def _speak(self, text: str):
        """TTS + playback, suppressing STT during output."""
        self.is_speaking = True
        try:
            audio = await text_to_speech(text)
            if audio:
                await self._send_audio(audio)
        except Exception as e:
            logger.error(f"Speak error: {e}", exc_info=True)
        finally:
            self.is_speaking = False
            # Flush any garbage transcript picked up during playback
            self.pending_transcript = ""
            await asyncio.sleep(0.3)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def handle_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    call = OperatorCall(reader, writer)
    try:
        await call.run()
    except Exception as e:
        logger.error(f"Call failed: {e}", exc_info=True)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("Call ended\n")


async def main():
    # Preflight checks
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
    logger.info("Dial 0 from the phone to connect.\n")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
