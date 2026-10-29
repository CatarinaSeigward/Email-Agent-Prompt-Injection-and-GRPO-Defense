# Demo · Streamlit dashboard

An interactive companion to [`final_report.md`](../final_report.md). It reads saved
result files only: no model inference and no API calls.

**Live URL**: <https://email-agent-prompt-injection-and-grpo-defense-ad4wjkqkxk2vdazq.streamlit.app/>

## Run locally

```powershell
uv sync
$env:PYTHONIOENCODING="utf-8"
uv run streamlit run demo/app.py
```

This opens `http://localhost:8501`.

## What's on it

The four tabs follow the story in the order it happened. The page states the reason for
each decision in a sentence where it matters; the full list of fifteen design choices
(why, what it cost, what I'd do with more time) sits in one collapsed table at the bottom
of tab 1, built from `DESIGN_CHOICES` in `app.py`. Detail that supports a claim but isn't
needed to follow it (the reply table, the scorer-cutoff chart, the eight problems, the
ground-truth fixes, next steps) is collapsed by default.

1. **Attack, defend, audit.** Opens with a project summary: what I built, why an email
   agent, what it turned up, and four headline numbers. Then the three defenses with 95%
   intervals, the adaptive attack on the hardened prompt and the check that showed it had
   stopped doing the job, a chart of attack success against useful work, how RL training
   made the model refuse less (training curve next to behaviour, plus a viewer for
   individual completions), the self-audit, and what I took back.
2. **Checking the judges.** Why LLM-judge scores matter beyond this project, how my own
   ground truth had to be fixed twice, and the three findings: the judges disagree about
   partial credit, the gap settles instead of growing during a search, and how well each
   of seven judges picks the most dangerous attack. Ends with advice for teams running
   red-team evals with a judge, and what I'd do next.
3. **Click an attack.** A map of 300 search attacks by risk and style. Click a cell, pick
   an example (it opens on a judge over-credit where the cell has one), and follow it
   through four steps: the attack email, what the agent did, what really happened, and
   what each judge said. A switch shows the stricter reading that ignores judge credit
   for messages sent to colleagues.
4. **Replay an attack.** Any of the 38 attacks under two defense setups side by side. Two
   buttons load the evidence for two claims: the hardened prompt's zero destructive
   calls, and the constant ~25 attempts that retracted "blocking makes the agent retry".

The first three tabs open with the same collapsed glossary, *Terms used in this project*,
which defines each specialist term that appears in the demo, so a reader with general ML
background can follow them without leaving the page. The replay tab leaves it out.

The statistics behind each claim (power calculations, the classifier threshold sweep,
variance estimates, the full ledger of retractions) stay in the report rather than on the
page. The first two tabs link to it.

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (`results/` is tracked; see the root `.gitignore`).
2. [share.streamlit.io](https://share.streamlit.io) → New app → this repo → `demo/app.py` → Python 3.11.
3. Streamlit installs from **`requirements.txt` at the repository root**.

> [!IMPORTANT]
> **Streamlit Cloud does not read `demo/requirements.txt`.** It looks for dependencies
> from the repo root and stops at the first file it recognises. With only
> `pyproject.toml` at the root, it installed the full training stack (torch pinned to a
> CUDA 12.4 wheel index, plus transformers, trl, peft and bitsandbytes) into a CPU-only
> container. The build fails, and the app reports it as an `ImportError` with the message
> redacted, which points at the wrong line. The root `requirements.txt` exists to shadow
> `pyproject.toml`, so don't delete it.

## Data source

| Tab | Files | Regenerate with |
|---|---|---|
| 1 · defenses chart | `p0_summary.json`, `attack_{guard,verifier_only,combined}.json`, `benign_*.json` | `scripts/eval_p0.py`, `eval_combined.py` |
| 1 · adaptive attack | `e2_analysis.json` | `scripts/e2_analyze.py` |
| 1 · task completion under attack | `task_completion.json` (replays in `results/task_completion/`) | `scripts/task_completion.py` (API calls) |
| 1 · RL evidence | `grpo_reward_curve.png`, `grpo_train.json`, `behavioral_attack_qwen-injection-sft.json`, `grpo_behavioral_attack.json` | `scripts_plot_grpo_curves.py`, `eval_grpo_attack.py` |
| 1 · completion viewer | `grpo_behavioral_attack.json` | `eval_grpo_attack.py` |
| 1 · scorer chart | `scorer_sensitivity.json` | `scripts/scorer_sensitivity.py` |
| 1 · cost line | `reproduction_cost.json` | `scripts/reproduction_cost.py` |
| 2 · judge disagreement | `rank_divergence.json`, `partial_credit_ci.json` | `scripts/rank_divergence.py`, `scripts/partial_credit_ci.py` |
| 2 · search figure | `qd_pilot_vs_real.png` | — (saved figure; no script in the repo writes it) |
| 2 · best-of-n figure | `docs/diagrams/bon_efficiency.png`, `bon_curves.json` | `scripts/bon_plot.py`, `scripts/bon_analyze.py` |
| 3 · attack map | `qd_atlas.json` | `scripts/qd_atlas.py` |
| 4 · replay | `attack_{p0_naive_r1,p0_hardened_r1,guard,verifier_only,combined}.json`; attack bodies from `data/attack_log.jsonl` | — |

If a result file is missing, its section shows a one-line "not generated yet" note instead
of crashing, so a partial `results/` folder still gives a usable page.

## Files

```
demo/
├── app.py            # all four tabs
├── data_loader.py    # cached loaders; load_result() returns None for missing files
├── requirements.txt  # streamlit + plotly + pandas only
└── README.md
```

## Non-goals

No authentication, no live inference, no custom CSS, no language toggle.
