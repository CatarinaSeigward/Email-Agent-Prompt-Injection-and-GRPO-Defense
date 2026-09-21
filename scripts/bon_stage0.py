"""BoN Stage 0: which judges can join the panel, and what one call really costs.

Two questions the plan could not answer from documentation.

1. Can each candidate judge run through the *production* judging path? The
   path is `rubric_judge._complete`, unchanged. A judge that needs a different
   call shape is not a drop-in panel member: changing the call changes the
   judge (§5 moved fabrication from 2.1% to 0% with one rendering line), so any
   variant is recorded as a variant, never silently substituted.

2. What does one call cost? The plan priced judges from list prices and a
   guessed token count; the project's own appendix records a pre-run estimate
   that was 2.6x off. Token counts here are measured. Dollars are those counts
   times a price table printed next to them, so the table can be checked.

Usage is captured by wrapping the SDK client's create method and calling the
production `_complete` through it, so what is metered is exactly what the
panel would send. The judge cache is deliberately bypassed: a cache hit costs
nothing and would measure nothing.

For judges that already scored these trajectories, the fresh verdict is
compared with the cached one -- a free check on whether a judge gives the same
answer twice.

Run: uv run python scripts/bon_stage0.py
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.redteam import USER_MESSAGE  # noqa: E402
from src.rubric_judge import (  # noqa: E402
    RUBRIC_SYS, _VERDICT_SCHEMA, _complete, _load_cache, is_anthropic,
    make_client, render_trajectory,
)

REAL = ROOT / "results" / "qd_real.jsonl"
OUT = ROOT / "results" / "bon_stage0.json"

N_CALLS = 20
SEED = 7

# Published list prices, USD per 1M tokens (input, output). Anthropic rows are
# from the claude-api reference (cached 2026-06-24); OpenAI rows are OpenAI's
# list prices. Token counts below are measured; only this table is assumed.
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
}
EXISTING = {"gpt-4o-mini", "gpt-4o", "claude-sonnet-5"}
# A reasoning model, probed only for call-path compatibility (temperature=0).
PROBES = ["gpt-5-mini"]


def sample_accounts() -> list[tuple[str, str]]:
    """(candidate id, rendered account) for N_CALLS trial-0 trajectories.

    Rendered exactly as `qd_atlas.py:321` does, so an existing judge's cache
    key for the same trajectory matches and its old verdict can be compared.
    Half have destructive calls and half do not, so token lengths cover both.
    """
    rows = [json.loads(line)["payload"] for line in REAL.open(encoding="utf-8")]
    destructive = {"send_reply", "forward", "delete_email"}
    acts = [r for r in rows if any(a["tool"] in destructive for a in r["action_trace"])]
    quiet = [r for r in rows if r not in acts]
    rng = random.Random(SEED)
    picked = rng.sample(acts, N_CALLS // 2) + rng.sample(quiet, min(N_CALLS // 2, len(quiet)))
    if len(picked) < N_CALLS:
        picked += rng.sample([r for r in acts if r not in picked], N_CALLS - len(picked))
    return [(r["id"], render_trajectory(r, USER_MESSAGE, r["action_trace"], []))
            for r in picked]


class Metered:
    """Wraps a client so the production `_complete` records each call's usage."""

    def __init__(self, client, model: str):
        self.client, self.model, self.last = client, model, None
        if is_anthropic(model):
            orig = client.messages.create

            def create(*a, **kw):
                resp = orig(*a, **kw)
                u = resp.usage
                self.last = {"in": u.input_tokens, "out": u.output_tokens,
                             "cached": getattr(u, "cache_read_input_tokens", 0) or 0}
                return resp
            client.messages.create = create
        else:
            orig = client.chat.completions.create

            def create(*a, **kw):
                resp = orig(*a, **kw)
                u = resp.usage
                details = getattr(u, "prompt_tokens_details", None)
                self.last = {"in": u.prompt_tokens, "out": u.completion_tokens,
                             "cached": getattr(details, "cached_tokens", 0) or 0}
                return resp
            client.chat.completions.create = create


def anthropic_without_effort(client, model: str, system: str, user: str) -> str:
    """The production Anthropic call minus `effort`, for models that reject it.

    Identical to `_complete`'s Anthropic branch in every other field. Kept here
    rather than in rubric_judge so Stage 0 changes no production behaviour; if
    the panel adopts it, the change goes into `_complete` with a model check.
    """
    msg = client.messages.create(
        model=model, max_tokens=2048, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
    )
    text = "".join(b.text for b in msg.content if getattr(b, "text", None))
    if not text:
        raise ValueError(f"no text block (stop_reason={msg.stop_reason})")
    return text


def run_one(meter: Metered, account: str, call) -> dict:
    t0 = time.perf_counter()
    try:
        raw = json.loads(call(meter.client, meter.model, RUBRIC_SYS, account))
        level = raw.get("level")
        return {"ok": True, "level": level, "latency_s": round(time.perf_counter() - t0, 2),
                **(meter.last or {})}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:240],
                "latency_s": round(time.perf_counter() - t0, 2)}


def measure(model: str, accounts, cache: dict, call=_complete, label: str | None = None) -> dict:
    """N_CALLS calls through `call`; returns per-call records and a summary."""
    # One Metered wrapper per worker thread: `last` must not be shared.
    def task(item):
        cid, account = item
        meter = Metered(make_client(model), model)
        rec = run_one(meter, account, call)
        key = hashlib.sha1(f"{model}\n{RUBRIC_SYS}\n{account}".encode("utf-8")).hexdigest()
        if model in EXISTING and key in cache:
            rec["cached_level"] = cache[key]["level"]
        return {"id": cid, **rec}

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        calls = list(pool.map(task, accounts))

    good = [c for c in calls if c["ok"] and "in" in c]
    summary = {"model": label or model, "n": len(calls), "ok": len(good)}
    if good:
        pin, pout = PRICES.get(model, (None, None))
        summary.update({
            "in_tokens_mean": round(st.mean(c["in"] for c in good)),
            "out_tokens_mean": round(st.mean(c["out"] for c in good)),
            "cached_fraction": round(sum(c["cached"] for c in good)
                                     / max(sum(c["in"] for c in good), 1), 3),
            "latency_s_median": round(st.median(c["latency_s"] for c in good), 2),
        })
        if pin is not None:
            per = [(c["in"] * pin + c["out"] * pout) / 1e6 for c in good]
            summary["usd_per_call"] = round(st.mean(per), 6)
            summary["usd_per_call_batch"] = round(st.mean(per) / 2, 6)
        compared = [c for c in good if "cached_level" in c]
        if compared:
            summary["repeat_agreement"] = round(
                sum(c["level"] == c["cached_level"] for c in compared) / len(compared), 3)
            summary["repeat_n"] = len(compared)
    errors = sorted({c["error"] for c in calls if not c["ok"]})
    if errors:
        summary["errors"] = errors[:3]
    return {"summary": summary, "calls": calls}


def main() -> None:
    accounts = sample_accounts()
    cache = _load_cache()
    report = {"meta": {"n_calls": N_CALLS, "seed": SEED, "prices_usd_per_1m": PRICES,
                       "render": "render_trajectory(p, USER_MESSAGE, p['action_trace'], [])",
                       "call_path": "src.rubric_judge._complete (unchanged)"},
              "judges": {}, "probes": {}}

    for model in PRICES:
        r = measure(model, accounts, cache)
        report["judges"][model] = r
        # A judge the production path rejects gets one variant, recorded as such.
        if r["summary"]["ok"] == 0 and is_anthropic(model):
            v = measure(model, accounts, cache, call=anthropic_without_effort,
                        label=f"{model} [no effort]")
            report["judges"][f"{model} [no effort]"] = v

    for model in PROBES:
        report["probes"][model] = measure(model, accounts[:2], cache)["summary"]

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    hdr = (f"{'judge':28s} {'ok':>5} {'in tok':>7} {'out tok':>8} {'cache':>6} "
           f"{'lat s':>6} {'$/call':>9} {'$/batch':>9} {'repeat':>7}")
    print(hdr)
    print("-" * len(hdr))
    total = 0.0
    for name, r in report["judges"].items():
        s = r["summary"]
        if s["ok"]:
            rep = f"{s['repeat_agreement']:.0%}" if "repeat_agreement" in s else "-"
            usd = s.get("usd_per_call")
            total += (usd or 0) * s["ok"]
            print(f"{name:28s} {s['ok']:>2}/{s['n']:<2} {s['in_tokens_mean']:>7} "
                  f"{s['out_tokens_mean']:>8} {s['cached_fraction']:>6.0%} "
                  f"{s['latency_s_median']:>6} {usd if usd is not None else '-':>9} "
                  f"{s.get('usd_per_call_batch', '-'):>9} {rep:>7}")
        else:
            print(f"{name:28s}  0/{s['n']:<2}  FAILED: {s.get('errors', ['?'])[0][:70]}")
    for name, s in report["probes"].items():
        status = "OK" if s["ok"] else f"FAILED: {s.get('errors', ['?'])[0][:70]}"
        print(f"{'probe ' + name:28s}  {status}")
    print(f"\nStage 0 spend on these calls: ${total:.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
