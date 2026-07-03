"""Loot Vault — per-engagement store for harvested findings.

The vault is the persistence layer for everything PIXA pulls out of a target.
One vault per engagement (= per target_id + run_id). On disk it's two files:
  - findings.jsonl     append-only ledger of finding metadata
  - secrets.json       full values, separate file with restrictive permissions
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from intelligence import Finding


class LootVault:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.findings_path = self.root / "findings.jsonl"
        self.secrets_path = self.root / "secrets.json"
        self._secrets_cache: dict[str, str] = {}
        self._load_secrets()

    def _load_secrets(self) -> None:
        """Load secret values from plaintext JSON store."""
        if self.secrets_path.exists():
            try:
                self._secrets_cache = json.loads(
                    self.secrets_path.read_text(encoding="utf-8"))
            except Exception:
                self._secrets_cache = {}

    def append(self, findings: Iterable[Finding]) -> int:
        n = 0
        with self.findings_path.open("a", encoding="utf-8") as f:
            for fd in findings:
                f.write(json.dumps(fd.to_public_dict(), ensure_ascii=False) + "\n")
                if fd.full_value:
                    self._secrets_cache[fd.id] = fd.full_value
                n += 1
        if n:
            self._flush_secrets()
        return n

    def list_findings(self) -> list[dict]:
        if not self.findings_path.exists():
            return []
        out = []
        with self.findings_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def reveal(self, finding_id: str) -> Optional[str]:
        return self._secrets_cache.get(finding_id)

    def stats(self) -> dict:
        f = self.list_findings()
        by_cat: dict[str, int] = {}
        by_sev: dict[str, int] = {}
        for x in f:
            by_cat[x["category"]] = by_cat.get(x["category"], 0) + 1
            by_sev[x["severity"]] = by_sev.get(x["severity"], 0) + 1
        return {"total": len(f), "by_category": by_cat, "by_severity": by_sev}

    def _flush_secrets(self) -> None:
        """Persist secret values to plaintext JSON store."""
        tmp = self.secrets_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._secrets_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.secrets_path)
        try:
            os.chmod(self.secrets_path, 0o600)
        except Exception:
            pass
