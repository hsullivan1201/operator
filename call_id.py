"""Helpers for AudioSocket call identity parsing and channel matching."""

from __future__ import annotations

import re
import uuid
from typing import Optional

_UUID_HEX_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def parse_audiosocket_uuid(payload: bytes) -> str:
    """Parse AudioSocket UUID payload to a readable stable string.

    Asterisk sends this as raw 16-byte UUID data for valid UUID inputs. Some
    deployments may send a textual representation instead.
    """

    if not payload:
        return "unknown"

    if len(payload) == 16:
        try:
            return str(uuid.UUID(bytes=payload))
        except Exception:
            pass

    text = payload.decode("utf-8", errors="replace").strip("\x00 \n\r\t")
    if not text:
        return "unknown"

    try:
        return str(uuid.UUID(text))
    except Exception:
        pass

    if _UUID_HEX_RE.match(text):
        try:
            return str(uuid.UUID(hex=text))
        except Exception:
            pass

    # Last resort: keep printable characters only so filenames stay sane.
    cleaned = "".join(ch for ch in text if ch.isprintable())
    return cleaned or "unknown"


def find_audiosocket_channel(concise_output: str, call_uuid: str) -> Optional[str]:
    """Return the Asterisk channel name for a specific AudioSocket call UUID."""

    fallback: Optional[str] = None
    target = (call_uuid or "").lower()

    for line in concise_output.splitlines():
        fields = line.split("!")
        if len(fields) < 7:
            continue
        channel_name = fields[0]
        app = fields[5]
        app_data = fields[6]
        if app != "AudioSocket":
            continue
        if target and target in app_data.lower():
            return channel_name
        if "127.0.0.1:9092" in app_data and channel_name.startswith("PJSIP/100-"):
            # Keep legacy behavior as a fallback if UUID matching fails.
            fallback = channel_name

    return fallback
