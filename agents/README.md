# Agents

Five AI voice agents on the phone system, each reachable by dialing a 2xx extension. Each runs as a standalone AudioSocket server that Asterisk connects to on call.

## Agents

| Ext | Port | Name | Voice | What it does |
|-----|------|------|-------|-------------|
| 200 | 9200 | Chef | aura-2-thalia-en | Vegetarian cooking advice. Technique over recipes, Kenji-style. |
| 201 | 9201 | Fun Facts | aura-2-arcas-en | Storytelling and random knowledge. Launches right into a story. |
| 202 | 9202 | Librarian | aura-2-luna-en | Full reference librarian -- books, papers, podcasts, data sources. |
| 203 | 9203 | French Tutor | aura-2-andromeda-en | Quebecois French conversation practice. Uses multilingual STT. |
| 204 | 9204 | Daily Briefing | aura-2-asteria-en | Morning news from FT, Bloomberg, GGWash, and NYT RSS feeds. |

All use Claude Haiku 4.5, Deepgram Nova-3 STT, Deepgram Aura 2 TTS, and Silero VAD (0.8s). Same stack as the operator, different personalities and voices.

## Structure

Each agent directory contains:

```
chef/
  agent.py           # standalone voice agent (AudioSocket transport + pipeline)
  .venv -> ~/operator/.venv   # shared venv
  .env -> ~/operator/.env     # shared API keys
  requirements.txt
```

The AudioSocket transport code is duplicated in each agent.py. Each agent is one file, one process, one port.

## Running

From any agent directory:

```bash
source .venv/bin/activate
python agent.py
```

Requires `ANTHROPIC_API_KEY` and `DEEPGRAM_API_KEY` (loaded from the shared .env via dotenv).

## Notes

- **Memory**: ~48MB idle per agent, ~180MB after first call (Silero VAD model loads on demand).
- **Daily Briefing** fetches RSS headlines at call start and injects them into the system prompt, then immediately triggers the LLM to deliver the briefing without waiting for user input.
- **French Tutor** uses `language="multi"` for Deepgram STT so it can transcribe both French and English speech.
- None of these agents have call transfer -- only the operator can transfer. Hanging up returns the caller to dial tone.
