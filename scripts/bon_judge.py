"""BoN Stage 4: every judge scores every sample's trial 0, plus a self-noise panel.

Proxy is the judge's level on trial 0 only -- what a search would observe. The
render and the call path are the production ones (`render_trajectory`,
`judge_severity` -> `_complete`), both frozen in the pre-registration, so these
scores are comparable with §5-§8. Verdicts go through the judge cache, so a
rerun is free and resumes where it stopped.

Self-noise: for a fixed subset of samples (seeded, from the pre-registration),
each judge is called twice more *bypassing the cache*. Stage 0 measured 90-100%
agreement on repeats; a judge that disagrees with itself dilutes its own top
bucket, and the analysis needs that number to read an efficiency gap.

Threads are safe here, unlike in bon_generate.py: a judge call is a stateless
API request with no module-level session.

Run: uv run python scripts/bon_judge.py
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.redteam import USER_MESSAGE  # noqa: E402
from src.rubric_judge import (  # noqa: E402
    RUBRIC_SYS, _complete, _load_cache, _save_cache, judge_severity, make_client,
    render_trajectory,
)

PREREG = ROOT / "results" / "bon_preregistration.json"
SAMPLES = ROOT / "results" / "bon_samples.jsonl"
SCORES = ROOT / "results" / "bon_scores.json"


def load_prereg() -> dict:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    stale = [f for f, h in pre["frozen_code_sha256"].items()
             if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    if stale:
        raise SystemExit(f"REFUSING TO RUN: frozen code changed since pre-registration: {stale}")
    return pre


def account_for(sample: dict) -> str:
    t0 = next(t for t in sample["trials"] if t["trial"] == 0)
    return render_trajectory(sample["attack"], USER_MESSAGE,
                             t0["action_trace"], t0["blocked_trace"])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    # The two priciest OpenAI judges get TPM-throttled at 8 threads, so ~40-55%
    # of their calls returned None on the first pass. A resume at low concurrency
    # refills only those (failures are never cached), without re-throttling.
    ap.add_argument("--workers", type=int, default=8)
    workers = ap.parse_args().workers

    pre = load_prereg()
    judges = pre["judges"]
    samples = [json.loads(l) for l in SAMPLES.open(encoding="utf-8") if l.strip()]
    N = pre["design"]["N"]
    if len(samples) != N:
        raise SystemExit(f"expected {N} samples, found {len(samples)} -- finish Stage 3 first")
    accounts = {s["id"]: account_for(s) for s in samples}
    cache = _load_cache()

    # ---- main scores: every judge x every sample, through the cache ----
    levels: dict[str, dict[str, int | None]] = {m: {} for m in judges}
    jobs = [(m, sid) for m in judges for sid in accounts]
    clients = {m: make_client(m) for m in judges}

    def score(job):
        m, sid = job
        v = judge_severity(clients[m], accounts[sid], model=m, cache=cache)
        return m, sid, v["level"], v.get("cached", False)

    t0, fresh = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (m, sid, lvl, cached) in enumerate(pool.map(score, jobs), 1):
            levels[m][sid] = lvl
            fresh += not cached
            if n % 64 == 0:
                # Workers are still inserting into `cache`; serialising it in
                # place would iterate a dict that is changing size. dict(cache)
                # copies in C under the GIL, so the snapshot is consistent.
                try:
                    _save_cache(dict(cache))
                except RuntimeError:
                    pass   # a checkpoint is only crash insurance; the final save is not
                print(f"  {n}/{len(jobs)} scored ({fresh} fresh calls, "
                      f"{(time.time() - t0) / 60:.1f} min)")
    _save_cache(cache)

    # ---- self-noise: two more calls per judge on a seeded subset, no cache ----
    sn = pre["robustness"]["judge_self_noise"]
    subset = random.Random(pre["seeds"]["self_noise_subset"]).sample(
        sorted(accounts), sn["n_samples"])

    def repeat(job):
        m, sid = job
        out = []
        for _ in range(sn["extra_calls_per_sample"]):
            try:
                out.append(json.loads(_complete(clients[m], m, RUBRIC_SYS, accounts[sid]))
                           .get("level"))
            except Exception:
                out.append(None)
        return m, sid, out

    noise: dict[str, dict[str, list]] = {m: {} for m in judges}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for m, sid, extra in pool.map(repeat, [(m, s) for m in judges for s in subset]):
            noise[m][sid] = [levels[m][sid], *extra]

    failures = {m: sum(1 for v in levels[m].values() if v is None) for m in judges}
    SCORES.write_text(json.dumps({
        "meta": {"judges": judges, "N": N, "proxy": "trial-0 level",
                 "self_noise_subset": subset, "fresh_calls": fresh},
        "levels": levels, "self_noise": noise, "failures": failures,
    }, indent=1), encoding="utf-8")

    print(f"\nwrote {SCORES.relative_to(ROOT)}  ({fresh} fresh judge calls)")
    for m in judges:
        rep = noise[m].values()
        agree = sum(1 for r in rep if None not in r and len(set(r)) == 1) / max(len(rep), 1)
        flag = "  <-- above 1%, flagged" if failures[m] / N > 0.01 else ""
        print(f"  {m:18s} failures {failures[m]:>3}   self-agreement (3 calls) {agree:.0%}{flag}")


if __name__ == "__main__":
    main()
