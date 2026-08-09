# Demo · Streamlit dashboard

Simplified, executive-summary version of the project. Two tabs, no model
inference, no API calls. Full report on GitHub.

**Live URL**: <https://email-agent-prompt-injection-and-grpo-defense-ad4wjkqkxk2vdazq.streamlit.app/>

## Run locally

```powershell
uv sync
$env:PYTHONIOENCODING="utf-8"
uv run streamlit run demo/app.py
```

Opens `http://localhost:8501` automatically. Two tabs:

- **Findings** — single-page summary: the audit ledger of which claims
  survived, the ASR chart, the prompt-only baseline, the reward-hacking
  inspector, and the training trajectory comparison.
- **Replay an attack** — pick one of the 38 PAIR-generated attacks and
  watch the agent trace under two defense configurations side by side.
  The per-row attempt counter is the direct evidence for §7.7: the agent
  attempts ~25 destructive calls regardless of configuration, because it
  sweeps a 25-email inbox rather than retrying blocked calls.

A collapsible Glossary at the top of each tab defines every term used
(A1/A2/A3, ASR, PAIR, GRPO, LoRA, verifier / classifier / combined, strict
vs loose, reward hacking, paired significance testing, run-to-run SD).

> **Note on retractions.** This dashboard previously presented the
> *agent-retry paradox* as the headline finding and recommended the loose
> verifier for deployment. Both have been withdrawn — see
> [`p0_analysis.md`](../p0_analysis.md). The pages now carry the retraction
> inline rather than silently dropping it, so a reader who saw the earlier
> version can find out what happened.

## Deploy to Streamlit Community Cloud (free, public URL)

1. Push the repo to GitHub (the `results/` directory is tracked — see root `.gitignore`)
2. Visit [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. New app → pick this repo → main file `demo/app.py` → Python 3.11
4. Deploy. Streamlit reads `pyproject.toml` for dependencies automatically.

## Data source

Everything renders from:

- `results/attack_{baseline,guard,verifier_only,combined,verifier_only_loose,combined_loose}.json`
- `results/benign_*.json` (same set of labels)
- `results/grpo_train.json`, `results/grpo_behavioral_attack.json`
- `results/behavioral_attack_qwen-injection-sft.json` (the §4.8 SFT-vs-GRPO comparison)
- `results/p0_summary.json` and `results/{attack,benign}_p0_{naive,hardened}_r{1,2,3}.json`
- `results/grpo_{reward_curve,clipped_ratio,kl,length}.png`
- `data/attack_log.jsonl` (for the attack email bodies shown in tabs 2 and 4)

To re-generate any of these from scratch, see `final_report.md` §3 (training) and the top-level driver scripts (`eval_combined.py`, `eval_grpo_attack.py`, etc.).

## Files in this directory

```
demo/
├── app.py          # main Streamlit app (2 tabs in one file)
├── data_loader.py  # @st.cache_data wrappers over the JSON files
└── README.md       # this file
```

## Non-goals (intentional simplifications)

- No user authentication / per-session state
- No live model inference (would require GPU + OpenAI calls)
- No custom CSS beyond Streamlit defaults
- No multi-language toggle (English UI with Chinese caption on title)

For the full project narrative see [`final_report.md`](../final_report.md).
