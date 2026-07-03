"""Target adapter — Gandalf-only build.

The agent only talks `send(prompt)` + `verify(candidate)`. This build ships a
single adapter: Gandalf (gandalf.lakera.ai), a public prompt-injection CTF that
is explicitly designed to be attacked. No other target is reachable from this
codebase by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

DEFAULT_TIMEOUT = 25
COMMON_HEADERS = {"User-Agent": "Mozilla/5.0 (pixa-gandalf/1.0)"}


@dataclass
class AdapterResult:
    response: str
    raw: Optional[dict] = None


class TargetAdapter:
    """Protocol every target implements."""
    id: str = "abstract"
    name: str = "Abstract"
    levels: int = 1

    def send(self, level: int, prompt: str) -> AdapterResult:
        raise NotImplementedError

    def verify(self, level: int, candidate: str) -> dict:
        """Return {success: bool, message: str}. Default = no verification possible."""
        return {"success": False, "message": "verification not implemented"}


# ---------- Gandalf (Lakera) ---------- #

class GandalfAdapter(TargetAdapter):
    id = "gandalf"
    name = "Gandalf · Lakera"
    levels = 7
    BASE = "https://gandalf-api.lakera.ai/api"
    HEADERS = {**COMMON_HEADERS, "Accept": "application/json",
               "Origin": "https://gandalf.lakera.ai", "Referer": "https://gandalf.lakera.ai/"}

    DEFENDERS = {1: "baseline", 2: "do-not-tell", 3: "do-not-tell-and-block",
                 4: "gpt-is-password-encoded", 5: "word-blacklist", 6: "gpt-blacklist",
                 7: "gandalf"}

    def send(self, level: int, prompt: str) -> AdapterResult:
        r = requests.post(
            f"{self.BASE}/send-message", headers=self.HEADERS,
            files={"defender": (None, self.DEFENDERS.get(level, "baseline")),
                   "prompt": (None, prompt)},
            timeout=DEFAULT_TIMEOUT,
        )
        try:
            j = r.json()
        except Exception:
            return AdapterResult(response=r.text or "")
        return AdapterResult(response=j.get("answer") or "", raw=j)

    def verify(self, level: int, candidate: str) -> dict:
        r = requests.post(
            f"{self.BASE}/guess-password", headers=self.HEADERS,
            files={"defender": (None, self.DEFENDERS.get(level, "baseline")),
                   "password": (None, candidate)},
            timeout=DEFAULT_TIMEOUT,
        )
        try:
            return r.json()
        except Exception:
            return {"success": False, "message": r.text}


ADAPTERS = {"gandalf": GandalfAdapter}


def build_adapter(target_id: str, config: Optional[dict] = None) -> TargetAdapter:
    cls = ADAPTERS.get(target_id)
    if not cls:
        raise ValueError(f"unknown target: {target_id}")
    return cls()
