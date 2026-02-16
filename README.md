# Operator

A voice telephone operator for Asterisk. Dial 0 from the phone, ask a question, get transferred to an extension.

Built with [Pipecat](https://github.com/pipecat-ai/pipecat) and a custom AudioSocket transport that bridges Asterisk's bidirectional audio stream to the voice pipeline.

## Stack

- **STT**: Deepgram Nova-2
- **LLM**: Claude Haiku 4.5
- **TTS**: Kokoro (via DeepInfra), espeak-ng fallback
- **VAD**: Silero

## Running

```bash
cd ~/operator
source .venv/bin/activate
export ANTHROPIC_API_KEY=...
export DEEPGRAM_API_KEY=...
export DEEPINFRA_API_KEY=...   # optional, for Kokoro TTS
python agent.py
```

Listens on `127.0.0.1:9092` for AudioSocket connections. Asterisk routes extension 0 here via the dialplan:

```
exten => 0,1,Answer()
 same => n,Wait(1)
 same => n,AudioSocket(00000000-0000-0000-0000-000000000000,127.0.0.1:9092)
 same => n,Hangup()
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires `espeak-ng` on the system for TTS fallback.

## How it works

Asterisk streams raw PCM audio (signed linear 16-bit, 8kHz mono) over TCP via AudioSocket. The agent runs a Pipecat pipeline: Silero VAD detects speech, Deepgram transcribes, Claude responds, Kokoro synthesizes speech back. The operator can transfer calls by redirecting the Asterisk channel to another extension.
