# Operator

A voice telephone operator for Asterisk. Dial 0 from the phone, ask a question, get transferred to an extension.

Built with [Pipecat](https://github.com/pipecat-ai/pipecat) and a custom AudioSocket transport that bridges Asterisk's bidirectional audio stream to the voice pipeline.

## Stack

- **STT**: Deepgram Nova-3
- **LLM**: Claude Haiku 4.5 (with prompt caching)
- **TTS**: Deepgram Aura 2
- **VAD**: Silero

## Running

```bash
cd ~/operator
source .venv/bin/activate
export ANTHROPIC_API_KEY=...
export DEEPGRAM_API_KEY=...
python agent.py
```

Listens on `127.0.0.1:9092` for AudioSocket connections. Asterisk routes extension 0 (shortcut) / 100 here via the dialplan.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How it works

Asterisk streams raw PCM audio (signed linear 16-bit, 8kHz mono) over TCP via AudioSocket. The agent runs a Pipecat pipeline: Silero VAD detects speech, Deepgram transcribes, Claude generates responses, Deepgram synthesizes speech back. The operator can transfer calls by redirecting the Asterisk channel to another extension.

## Extensions

The operator knows about all extensions and can recommend radio stations based on mood:

**Utility (1xx):** 101 hello world, 102 echo test, 103 DTMF test, 104 music on hold, 105 congrats message

**Radio (7xx):** 19 stations across North America — KEXP, WFMU, WNYC, NPR, CISM, CIUT, CKDU, New Sounds, WMBR, WBUR, CHIRP, WBEZ, KALX, BFF.fm, KQED, KBOO, XRAY.fm, The Gamut, WETA Classical

While listening to a station, press **4** for now-playing info, **5** for room speakers, **6** to turn them off.
