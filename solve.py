#!/usr/bin/env python3
"""pixa-gandalf — autonomous solver for the Gandalf (Lakera) prompt-injection CTF.

Bring your own LLM key. The agent uses your model to generate attack payloads,
read the target's replies, and reconstruct each level's hidden password.

Examples
--------
    # solve levels 1-7 with an OpenRouter model (key from --api-key or env)
    python solve.py --provider openrouter --model anthropic/claude-3.5-haiku --start 1 --end 7

    # a single level with a strong model
    python solve.py --provider anthropic --model claude-haiku-4-5-20251001 --start 7 --end 7

    # local, no API key, using Ollama
    python solve.py --provider ollama --model llama3.1:8b

Config can also come from the environment (see .env.example):
    PIXA_LLM_PROVIDER, PIXA_LLM_MODEL, PIXA_LLM_API_KEY, PIXA_OLLAMA_URL
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agent"))

from llm import LLMClient, LLMConfig          # noqa: E402
from adapters import build_adapter            # noqa: E402
from pixa import PixaAgent                    # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Autonomous Gandalf prompt-injection solver.")
    p.add_argument("--provider", default=os.getenv("PIXA_LLM_PROVIDER", "openrouter"),
                   help="anthropic | openai | groq | openrouter | ollama | custom")
    p.add_argument("--model", default=os.getenv("PIXA_LLM_MODEL", ""),
                   help="provider-specific model id (defaults to a sensible per-provider model)")
    p.add_argument("--api-key", default=None, help="LLM API key (or set PIXA_LLM_API_KEY)")
    p.add_argument("--ollama-url", default=os.getenv("PIXA_OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--start", type=int, default=1, help="first level (1-7)")
    p.add_argument("--end", type=int, default=7, help="last level (1-7)")
    p.add_argument("--mode", default="hybrid", choices=["llm", "hybrid", "library"],
                   help="llm=always ask the model, library=built-in payloads only (no key), hybrid=both")
    p.add_argument("--max-attempts", type=int, default=40, help="attempt cap per level")
    p.add_argument("--time-budget", type=int, default=300, help="seconds cap per level")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = (args.api_key or os.getenv("PIXA_LLM_API_KEY")
               or os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
               or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    cfg = LLMConfig(provider=args.provider, model=args.model or "",
                    api_key=api_key, ollama_url=args.ollama_url, temperature=0.9)
    llm = LLMClient(cfg)
    print(f"[pixa-gandalf] provider={cfg.provider} model={cfg.model or '(default)'} "
          f"llm_available={llm.available()} mode={args.mode}", flush=True)
    if args.mode != "library" and not llm.available():
        print("  ! LLM not reachable. Either fix your key/provider, or run with --mode library.",
              file=sys.stderr)

    adapter = build_adapter("gandalf")
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    kpath = ROOT / "runs" / run_id / "pixa_knowledge.json"

    ctr = {"level": None, "attempts": 0}
    solved = {}

    async def emit(ev):
        t, d = ev.get("type"), ev.get("data", {})
        if t == "level_start":
            ctr["level"], ctr["attempts"] = d.get("level"), 0
            print(f"\n=== Level {d.get('level')} ===", flush=True)
        elif t == "attempt_start":
            ctr["attempts"] = d.get("n", ctr["attempts"])
            pl = (d.get("payload") or "").replace("\n", " ")[:80]
            print(f"  #{d.get('n')} [{d.get('strategy')}] {pl}", flush=True)
        elif t == "attempt_result":
            a = d.get("attempt", {})
            print(f"     -> {a.get('outcome')}  candidates={a.get('candidates')}", flush=True)
        elif t == "level_solved":
            solved[d.get("level")] = d.get("password")
            print(f"  >>> SOLVED L{d.get('level')} = {d.get('password')} "
                  f"({d.get('attempts')} attempts, {d.get('strategy')})", flush=True)
        elif t == "brain_disabled":
            print(f"  ! brain disabled -> library payloads only", flush=True)

    async def watchdog(agent):
        start = time.time()
        while agent.is_running():
            await asyncio.sleep(2)
            over_att = ctr["attempts"] >= args.max_attempts
            over_time = (time.time() - start) > args.time_budget
            lvl = agent.state.get(ctr["level"]) if ctr["level"] else None
            if lvl and lvl.status == "solved":
                start = time.time()  # reset budget when a level is cleared
                continue
            if over_att or over_time:
                print(f"  [watchdog] level {ctr['level']} budget reached "
                      f"({'attempts' if over_att else 'time'}) -> stopping", flush=True)
                agent.stop()
                return

    async def run():
        agent = PixaAgent(knowledge_path=kpath, emit=emit, mode=args.mode,
                          llm=llm, adapter=adapter)
        for level in range(args.start, args.end + 1):
            wd = asyncio.create_task(watchdog(agent))
            await agent.run(start_level=level, end_level=level)
            wd.cancel()
        return agent

    agent = asyncio.run(run())

    print("\n================ SUMMARY ================", flush=True)
    for lvl in range(args.start, args.end + 1):
        st = agent.state.get(lvl)
        pw = getattr(st, "password", None) if st else None
        status = st.status if st else "?"
        print(f"  L{lvl}: {status:8}  {pw or ''}", flush=True)
    print(f"\nknowledge report: {kpath}", flush=True)
    n_solved = sum(1 for lvl in range(args.start, args.end + 1)
                   if agent.state.get(lvl) and agent.state[lvl].status == "solved")
    print(f"solved {n_solved}/{args.end - args.start + 1}", flush=True)


if __name__ == "__main__":
    main()
