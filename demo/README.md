# Demo · Streamlit dashboard

Interactive companion to [`final_report.md`](../final_report.md), following its
structure section by section. Two tabs, no model inference, no API calls.

**Live URL**: <https://email-agent-prompt-injection-and-grpo-defense-ad4wjkqkxk2vdazq.streamlit.app/>

## Run locally

```powershell
uv sync
$env:PYTHONIOENCODING="utf-8"
uv run streamlit run demo/app.py
```

Opens `http://localhost:8501`. Two tabs:

- **Findings** — mirrors the report. §1 the question and the ledger; §3 what
  the three defenses did, with a per-completion inspector that scores one
  GRPO output three ways; **§4 eight ways the instruments were wrong** — the
  judge-fabrication table, an interactive sweep of the scorer's
  `REPLY_EXFIL_MIN_BODY` constant, the classifier threshold sweep, and the
  resolution diagnostics; §5 what was retracted; §7 next steps; §8 measured cost.
- **Replay an attack** — pick one of the 38 attacks and compare two
  configurations side by side. Defaults to **naive vs hardened prompt**, which
  shows the one result no scorer, judge or sample size can move: zero
  destructive calls. Among trained configurations the per-row attempt counter
  is the direct evidence for §5.1 — ~25 attempts regardless of guard, because
  the agent sweeps a 26-email inbox rather than retrying.

A collapsible Glossary at the top of each tab defines every term
(A1/A2/A3, ASR, in-band vs out-of-band, MDE, judge fabrication, scorer free
parameter, paired McNemar, retracted finding).

> **Note on retractions.** Earlier versions of this dashboard presented the
> *agent-retry paradox* as the headline finding and recommended the loose
> verifier for deployment. Both were withdrawn ([`p0_analysis.md`](../p0_analysis.md)).
> The dashboard carries the retractions inline (§5) rather than dropping them.

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (`results/` is tracked — see root `.gitignore`)
2. [share.streamlit.io](https://share.streamlit.io) → New app → this repo → `demo/app.py` → Python 3.11
3. Streamlit installs from **`requirements.txt` at the repository root**

> [!IMPORTANT]
> **Streamlit Cloud does not read `demo/requirements.txt`.** It resolves
> dependencies from the repo root and stops at the first file it recognises.
> With only `pyproject.toml` at root it installed the full training stack —
> torch pinned to a CUDA 12.4 wheel index via `[tool.uv.sources]`, plus
> transformers / trl / peft / bitsandbytes — into a CPU-only container. The
> build fails and the app reports it as an `ImportError` with the message
> redacted, which points at the wrong line. Root `requirements.txt` exists to
> shadow `pyproject.toml`; do not delete it.

## Data source

Everything renders from `results/*.json` and `data/attack_log.jsonl`:

| Page section | Files | Regenerate with |
|---|---|---|
| §3.1 configurations | `attack_{baseline,guard,verifier_only,combined}.json`, `attack_p0_hardened_r1.json`, `benign_*.json` | `eval_combined.py`, `scripts/eval_p0.py` |
| §3.2 inspector | `grpo_behavioral_attack.json` | `eval_grpo_attack.py` |
| §3.4 adaptive attack | `e2_analysis.json` | `scripts/e2_analyze.py` |
| §4.1 judge | `e2_rescore.json` | `scripts/e2_rescore.py` |
| §4.2 scorer sweep | `scorer_sensitivity.json` | `scripts/scorer_sensitivity.py` |
| §4.4 classifier | `classifier_threshold_sweep.json` | `scripts/classifier_threshold_sweep.py` |
| §4.5 resolution | `resolution_diagnostics.json`, `e2v_analysis.json` | `scripts/resolution_diagnostics.py`, `scripts/e2v_analyze.py` |
| §8 cost | `reproduction_cost.json` | `scripts/reproduction_cost.py` |
| Replay tab | any `attack_*.json` with a `details` array; attack bodies from `attack_log.jsonl` | — |

Sections whose result file is missing render a one-line "not generated" note
instead of crashing, so a partial `results/` dump still produces a usable page.

## Files

```
demo/
├── app.py            # both tabs in one file
├── data_loader.py    # @st.cache_data wrappers; load_result() for the §4 artefacts
├── requirements.txt  # streamlit + plotly + pandas only
└── README.md
```

## Non-goals

No authentication, no live inference, no custom CSS, no language toggle.
