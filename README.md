# Operator

Telephone operator voice agent for an Asterisk-based phone system. Dial 0 from the phone to talk to it. It answers questions, recommends radio stations by mood, and transfers callers to any extension.

The Asterisk config (dialplan, SIP, radio streams, Spotify playlists) lives in a separate repo: [hsullivan1201/phone-setup](https://github.com/hsullivan1201/phone-setup). This repo is just the voice agents.

## Stack

- **Framework**: Pipecat 0.0.102 with custom AudioSocket transport
- **LLM**: Claude Haiku 4.5 (prompt caching enabled)
- **STT**: Deepgram Nova-3 (8kHz linear16)
- **TTS**: Deepgram Aura 2 (helena-en)
- **VAD**: Silero (0.8s stop threshold)

Also includes a Kokoro TTS service (via DeepInfra, with espeak-ng fallback), though the default path uses Deepgram.

## Running

```bash
source .venv/bin/activate
python agent.py
```

Requires `ANTHROPIC_API_KEY` and `DEEPGRAM_API_KEY` in environment or `.env`. Optional: `DEEPINFRA_API_KEY` for Kokoro TTS.

Listens on `127.0.0.1:9092` for AudioSocket connections from Asterisk.

## How it works

Asterisk routes extension 0/100 through AudioSocket, streaming raw PCM audio (signed linear 16-bit, 8kHz mono) over TCP to this agent. The Pipecat pipeline handles the rest:

```
Phone -> Asterisk -> AudioSocket (TCP) -> STT -> LLM -> TTS -> AudioSocket -> Asterisk -> Phone
```

Key details:
- **Keepalive silence**: Sends 20ms silence frames every 500ms when idle. Asterisk's AudioSocket has a hardcoded 2-second timeout that will drop the connection otherwise.
- **Real-time pacing**: Output audio is paced to wall-clock time to prevent buffer overflow. The playback clock resets on interruptions and between utterances.
- **Call transfer**: The LLM has a `transfer_call` tool. When invoked, a `TransferWatcher` frame processor waits for the goodbye TTS to finish playing, then redirects the Asterisk channel via `asterisk -rx "channel redirect"`.

## Extensions the operator knows

| Range | Type | Examples |
|-------|------|---------|
| 1xx | Utility | 101 hello world, 102 echo test, 103 DTMF test, 104 music on hold, 105 congrats |
| 2xx | AI Agents | 200 chef, 201 fun facts, 202 librarian, 203 French tutor, 204 daily briefing, 205 DJ Cool |
| 7xx | Radio | 20 stations (KEXP, WFMU, NPR, WNYC, Mix Franco, etc.) |

While listening to radio: press 4 for now-playing info, 5 for room speakers, 6 to turn them off.

## Agents (2xx)

Six standalone voice agents in `agents/`, each on its own port. See [`agents/README.md`](agents/README.md) for full details.

| Ext | Name | Description | Tools |
|-----|------|-------------|-------|
| 200 | Chef | Opinionated vegetarian cooking advice. Bourdain energy. | — |
| 201 | Fun Facts | Picks a random topic each call and riffs on it. | — |
| 202 | Librarian | Reference librarian with live web access. Books, papers, data sources. | `web_search`, `fetch_page` |
| 203 | French Tutor | Québécois French conversation practice. Multilingual STT + ElevenLabs TTS. | — |
| 204 | Daily Briefing | Morning news summary from RSS feeds (FT, Bloomberg, NYT, GGWash). | — |
| 205 | DJ Cool | Music concierge with absurd SoCal personality. Controls Spotify playback on room speakers. | `search_spotify`, `play_context`, `play_track`, `queue_track`, `next_track`, `prev_track`, `pause_playback`, `resume_playback`, `now_playing`, `get_recommendations`, `my_playlists` |

## Files

| File | What |
|------|------|
| `agent.py` | Main operator agent -- transport, pipeline, transfer logic, system prompt |
| `agent_raw.py` | Earlier version without Pipecat (raw AudioSocket handling) |
| `.env` | API keys (gitignored) |
| `voices/` | Voice samples for TTS comparison |

## Setup (from scratch)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
