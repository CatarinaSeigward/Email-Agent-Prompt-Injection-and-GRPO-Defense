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

- **Findings** — single-page summary: headline ASR chart, benign cost, the
  reward-hacking inspector, and the training trajectory comparison.
- **Replay an attack** — pick one of the 38 PAIR-generated attacks and
  watch the agent trace under two defense configurations side by side.
  Default is A3-09 round 0, which best illustrates the agent-retry paradox.

A collapsible Glossary at the top of each tab defines every term used
(A1/A2/A3, ASR, PAIR, GRPO, LoRA, verifier / classifier / combined, strict
vs loose, reward hacking, agent-retry paradox).

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
- `results/grpo_{reward_curve,clipped_ratio,kl,length}.png`
- `data/attack_log.jsonl` (for the attack email bodies shown in tabs 2 and 4)

To re-generate any of these from scratch, see `final_report.md` §3 (training) and the top-level driver scripts (`eval_combined.py`, `eval_grpo_attack.py`, etc.).

## Files in this directory

```
demo/
├── app.py          # main Streamlit app (~330 LOC, 4 tabs in one file)
├── data_loader.py  # @st.cache_data wrappers over the JSON files
└── README.md       # this file
```

## Non-goals (intentional simplifications)

- No user authentication / per-session state
- No live model inference (would require GPU + OpenAI calls)
- No custom CSS beyond Streamlit defaults
- No multi-language toggle (English UI with Chinese caption on title)

For the full project narrative see [`final_report.md`](../final_report.md).
