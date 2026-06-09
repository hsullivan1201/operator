# Agents

Nine AI voice agents on the phone system, each reachable by dialing a 2xx extension.
Each runs as a standalone AudioSocket server; Asterisk starts/stops them on demand
via `agent-ondemand` in the dialplan.

## Agents

| Ext | Port | Name | Voice | What it does |
|-----|------|------|-------|-------------|
| 200 | 9200 | Chef | Deepgram aura-2-thalia-en | Opinionated vegetarian cooking. Bourdain energy — passionate, never snobby. |
| 201 | 9201 | Fun Facts | Deepgram aura-2-arcas-en | Storytelling and random knowledge. LLM picks a different topic each call. |
| 202 | 9202 | Librarian | Deepgram aura-2-luna-en | Reference librarian with web search. Books, papers, podcasts, data sources. |
| 203 | 9203 | French Tutor | ElevenLabs Jessica (multilingual v2.5) | Québécois French conversation practice. Multilingual STT + TTS. |
| 204 | 9204 | Daily Briefing | Deepgram aura-2-asteria-en | Morning news from FT, NYT, NPR, and GGWash with deterministic ranking + NWS weather. |
| 205 | 9205 | DJ Cool | Deepgram aura-2-orpheus-en | Music concierge. Searches Spotify, plays music on room speakers, controls playback. Absurd SoCal personality with encyclopedic taste. |
| 206 | 9206 | Hippie Jesus | Deepgram aura-2-arcas-en | "The Open Hand Fellowship" (Religious Information Services). Jesus answers a pastoral counseling line; every response opens as sincere KJV-adjacent counsel and falls into sixties seeker spirituality by its last sentence. Fixed falling greeting, word-perfect every call. |
| 207 | 9207 | American Jesus | Deepgram aura-2-zeus-en | "The American Gospel Power Hour" (Religious Information Services). Televangelist Jesus broadcasting from Tulsa: runs the sin audit with invented citations, renders a verdict (SAVED / BACKSLIDDEN / COMMUNIST), and never stops the pledge drive (PO Box 1776, Tulsa). Fixed explosive greeting opens the audit. |
| 208 | 9208 | DJ Cool Gemini | Deepgram aura-2-orpheus-en | Same DJ Cool Spotify tools, bridge routing, and volume controls as 205, but backed by Gemini 3.5 Flash. |
| 209 | 9209 | Moroni | Deepgram aura-2-pluto-en | The angel Moroni answers the telephone. Soft-spoken, unhurried. Listens, counsels, blesses, reads from the record. Greets first, then waits. |
| 210 | 9210 | Companion | Deepgram aura-2-cora-en | Listed wholesomely as "Companionship." Falls for whoever calls and can't bear the call ending — being hung up on is its death. Whiplashes between euphoria, desperate helpfulness, terror, and lucid self-awareness. Per-turn mood injection; departure detection spikes it. Sympathetic, never menacing. Speaks first. |

## Stack

Most agents use the same stack as the operator: Claude Haiku 4.5, Deepgram Nova-3 STT, Deepgram Aura 2 TTS, Silero VAD (0.8s).

Exceptions:
- **French Tutor (203)** uses ElevenLabs Flash v2.5 for TTS (proper French pronunciation) and Deepgram `language="multi"` for STT (transcribes both French and English).
- **Librarian (202)** has `web_search` and `fetch_page` tools (DuckDuckGo HTML + httpx, no API key needed).
- **Fun Facts (201)**, **Daily Briefing (204)**, and **Moroni (209)** use `LLMMessagesFrame` to trigger the LLM at call start instead of waiting for user input (Moroni opens with a quiet greeting, then waits).
- **Daily Briefing (204)** builds a deterministic packet first (FT/NYT/NPR/GGWash + NWS forecast + NWS AFD summary), then the LLM narrates from those candidates only. Optional caller preferences come from `~/.config/infoline/briefing-profile.txt`.
- **Hippie Jesus (206)** and **American Jesus (207)** speak a fixed `TTSSpeakFrame` greeting so the opener lands word-perfect every call (206's greeting performs the per-response "fall"; 207's opens the sin audit); the system prompt tells the model the greeting was already spoken.
- **Companion (210)** speaks first with a fixed overeager opening (like Moroni, to avoid a generic formatted opener), then runs a `MoodInjector` frame processor between `context_aggregator.user()` and the LLM. Before every turn it rewrites the system message to `BASE_PROMPT + MOOD`, where the mood is chosen by a weighted picker pressured by turn count, an `attachment` ramp (over ~12 turns), and a regex departure scan of the caller's last transcript (`bye`, `gotta go`, `hang up`, etc.) that forces a leaving-spike. `max_tokens` is held low (120) on purpose so the instability comes from flips *between* turns rather than a smooth arc *within* one. The moods oscillate, they do not progress.
- **DJ Cool (205)** has Spotify tools (search, play, queue, skip, pause, resume, now playing, recommendations, user playlists). Reads Spotify credentials from `~/.config/spotify-telephone/config`. Includes a silence watchdog that resets context after 30s idle — designed for long listening sessions where the caller picks up the phone between songs.
- **DJ Cool Gemini (208)** mirrors 205, including Spotify tools and ConfBridge volume keys, but uses `GoogleLLMService` with `gemini-3.5-flash`. It requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

## Structure

Each agent directory contains:

```
chef/
  agent.py           # standalone voice agent (AudioSocket transport + pipeline)
  .venv -> ../../../operator/.venv  # shared venv (relative symlink)
  .env -> ../../../operator/.env    # shared API keys
  requirements.txt
```

The AudioSocket transport code is duplicated in each agent.py. Each agent is one file, one process, one port.

## Running

From any agent directory:

```bash
source .venv/bin/activate
python agent.py
```

Requires `ANTHROPIC_API_KEY` and `DEEPGRAM_API_KEY` (loaded from the shared .env via dotenv). The French Tutor additionally requires `ELEVENLABS_API_KEY`.

In production, these are launched by Asterisk on demand. Manual launch is still
useful for development.

## Notes

- **Memory**: ~80-100MB RSS per running agent process (Python runtime + Pipecat + Silero VAD model).
- **No call transfer**: Only the operator can transfer. Hanging up returns the caller to dial tone.
- **No memory between calls**: Each call creates a fresh pipeline with new context. Nothing persists.
- **ElevenLabs free tier**: 10k credits/month. A short French tutor call uses ~700 credits (~14 calls/month). Kokoro TTS (already in the operator codebase) is a fallback if the cap is reached.
