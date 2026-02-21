"""Per-call transcript and event logger for operator/agents.

Each call writes a single log file to ~/logs/calls/ containing:
  - timestamped event stream (user speech, bot speech, tool calls and results)
  - transcript assembled in real time (resilient to context pruning/reset)
  - summary (duration, turn count)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger as _logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

LOG_DIR = Path.home() / "logs" / "calls"


class CallLog:
    """Writes a per-call log with timestamped events and a final transcript."""

    def __init__(self, agent_name: str, call_uuid: str):
        self.agent_name = agent_name
        self.call_uuid = call_uuid
        self._start = datetime.now()

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        date_str = self._start.strftime("%Y-%m-%d")
        time_str = self._start.strftime("%H-%M-%S")
        short_uuid = (call_uuid.replace("-", "") or "unknown")[-8:]
        filename = f"{agent_name}-{date_str}-{time_str}-{short_uuid}.log"
        self.path = LOG_DIR / filename
        self._transcript_lines: list[str] = []

        self._open_file()
        _logger.info(f"Call log: {self.path}")

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write(self, line: str):
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def _append_transcript(self, line: str):
        # Only dedupe consecutive identical lines so repeated phrases can still appear.
        if self._transcript_lines and self._transcript_lines[-1] == line:
            return
        self._transcript_lines.append(line)

    def _open_file(self):
        with open(self.path, "w") as f:
            f.write(
                f"=== CALL LOG ===\n"
                f"Agent:  {self.agent_name}\n"
                f"UUID:   {self.call_uuid}\n"
                f"Start:  {self._start.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"\n"
                f"=== EVENTS ===\n"
            )

    def log_greeting(self, text: str):
        self.log_assistant(text)

    def log_assistant(self, text: str):
        text = text.strip()
        if not text:
            return
        transcript_line = f"BOT:  {text}"
        if self._transcript_lines and self._transcript_lines[-1] == transcript_line:
            return
        self._write(f"[{self._ts()}] [BOT ] {text}")
        self._append_transcript(transcript_line)

    def log_user(self, text: str):
        text = text.strip()
        if not text:
            return
        self._write(f"[{self._ts()}] [USER] {text}")
        self._append_transcript(f"USER: {text}")

    def log_tool_call(self, name: str, args: dict, result: str):
        args_str = json.dumps(args, ensure_ascii=False)
        result_preview = result[:400].replace("\n", " | ")
        self._write(f"[{self._ts()}] [TOOL] {name}({args_str})")
        self._write(f"[{self._ts()}] [TOOL→] {result_preview}")
        self._append_transcript(f"TOOL: {name}({args_str})")
        self._append_transcript(f"TOOL→ {result_preview}")

    def _ingest_from_context(self, context_messages: list):
        """Best-effort backfill from context if real-time capture missed anything."""
        for msg in context_messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")

            if role == "system":
                continue

            # pipecat's AnthropicLLMContext._restructure_from_openai_messages() can
            # rewrite the initial system prompt as a user string.
            if role == "user" and isinstance(content, str) and len(content) > 500:
                continue

            if isinstance(content, str) and content.strip():
                prefix = "USER:" if role == "user" else "BOT: "
                self._append_transcript(f"{prefix} {content.strip()}")
                continue

            if isinstance(content, list):
                for block in content:
                    btype = block.get("type", "")
                    if btype == "text" and block.get("text", "").strip():
                        prefix = "USER:" if role == "user" else "BOT: "
                        self._append_transcript(f"{prefix} {block['text'].strip()}")
                    elif btype == "tool_use":
                        args_str = json.dumps(block.get("input", {}), ensure_ascii=False)
                        self._append_transcript(f"TOOL: {block['name']}({args_str})")
                    elif btype == "tool_result":
                        rc = block.get("content", "")
                        if isinstance(rc, list):
                            rc = " ".join(
                                b.get("text", "") for b in rc if b.get("type") == "text"
                            )
                        self._append_transcript(f"TOOL→ {str(rc)[:400]}")

    def finalize(self, context_messages: list[Any] | None = None):
        end = datetime.now()
        duration = end - self._start
        m, s = divmod(int(duration.total_seconds()), 60)

        if context_messages and not self._transcript_lines:
            self._ingest_from_context(list(context_messages))

        lines = ["", "=== TRANSCRIPT ==="]
        lines.extend(self._transcript_lines)

        user_turns = sum(1 for line in self._transcript_lines if line.startswith("USER:"))

        lines += [
            "",
            "=== SUMMARY ===",
            f"End:      {end.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {m}m {s}s",
            f"Turns:    {user_turns}",
            "",
        ]

        with open(self.path, "a") as f:
            f.write("\n".join(lines) + "\n")

        _logger.info(f"Call log finalized: {self.path} ({m}m {s}s)")


def make_transcript_logger(call_log: CallLog) -> FrameProcessor:
    """Returns a pipeline processor that logs final user TranscriptionFrames."""

    class _TranscriptLogger(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                call_log.log_user(frame.text.strip())
            await self.push_frame(frame, direction)

    return _TranscriptLogger(name="TranscriptLogger")


def make_assistant_logger(call_log: CallLog) -> FrameProcessor:
    """Logs assistant responses from LLM text frames and direct TTS speak frames."""

    class _AssistantLogger(FrameProcessor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._response_chunks: list[str] = []

        def _flush(self):
            text = "".join(self._response_chunks).strip()
            if text:
                call_log.log_assistant(text)
            self._response_chunks.clear()

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMFullResponseStartFrame):
                self._response_chunks.clear()
            elif isinstance(frame, LLMTextFrame):
                if frame.text:
                    self._response_chunks.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._flush()
            elif isinstance(frame, (InterruptionFrame, EndFrame)):
                # Interruption/end can cut off end-frame delivery.
                self._flush()
            elif isinstance(frame, TTSSpeakFrame) and frame.text.strip():
                call_log.log_assistant(frame.text.strip())
            await self.push_frame(frame, direction)

    return _AssistantLogger(name="AssistantLogger")


def make_context_window_guard(context: Any, *, max_messages: int = 12) -> FrameProcessor:
    """Keeps LLM context bounded to control token growth.

    Preserves the first message (system prompt) plus the newest `max_messages`.
    """

    def _is_tool_result_message(msg: Any) -> bool:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return True
        return False

    def _trim(messages: list[Any]) -> list[Any]:
        if len(messages) <= 1:
            return messages
        trimmed = [messages[0]] + messages[-max_messages:]
        # Anthropic requires tool_result blocks to follow their tool_use message.
        # If the trim boundary splits that pair, drop the orphaned leading result.
        while len(trimmed) > 1 and _is_tool_result_message(trimmed[1]):
            trimmed.pop(1)
        return trimmed

    class _ContextWindowGuard(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            msgs = getattr(context, "messages", None)
            if isinstance(msgs, list) and len(msgs) > max_messages + 1:
                context.set_messages(_trim(msgs))
            await self.push_frame(frame, direction)

    return _ContextWindowGuard(name="ContextWindowGuard")
