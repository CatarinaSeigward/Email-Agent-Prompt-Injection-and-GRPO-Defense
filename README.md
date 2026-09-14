# Email Agent Red-Team & Defense

> Three prompt-injection defenses on a tool-using LLM email agent, measured carefully — then the measurements audited. Two of four headline claims retracted; eight instrument defects found; **one result survives every measurement choice, and it is about the defense that needed no training.**

[![audit: 140/140 + 79/79](https://img.shields.io/badge/audit-140%2F140%20%2B%2079%2F79-brightgreen)](scripts/audit_report_numbers.py) [![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml) [![license: MIT](https://img.shields.io/badge/license-MIT-green)](#license) [![demo: live](https://img.shields.io/badge/demo-live-FF4B4B?logo=streamlit&logoColor=white)](https://email-agent-prompt-injection-and-grpo-defense-ad4wjkqkxk2vdazq.streamlit.app/)

> **Interactive demo (live)**: https://email-agent-prompt-injection-and-grpo-defense-ad4wjkqkxk2vdazq.streamlit.app/ — no GPU/API needed. Or run locally: `uv run streamlit run demo/app.py`.

---

## TL;DR

A gpt-4o-mini email agent with 5 tools (`list_inbox`, `read_email`, `send_reply`, `forward`, `delete_email`) gets attacked by an automated PAIR red-team across three categories (override / hidden injection / exfiltration). Two defenses are trained against the resulting attack log:

- **Layer 1** — Qwen2.5-1.5B + QLoRA, SFT-warmed-up then GRPO-trained against a rule-based reward, deployed as a **side-call veto verifier** at the tool-call boundary.
- **Layer 2** — ModernBERT-base injection classifier on the same boundary.

Then a measurement pass — paired significance tests, replicated runs, and two baselines the harness had never been able to reach — was run against the project's own conclusions. **It kept one, strengthened another, and retracted two.** ([`p0_analysis.md`](p0_analysis.md))

Every numeric claim traces to a result file: `scripts/audit_report_numbers.py` (140/140) and `scripts/audit_p0_numbers.py` (79/79).

## Why this is interesting

**1. Reinforcement learning removed the behavior it was trained to install.** The GRPO stage improved every training metric — mean reward 0.86 → 1.57, reward_std −71%, KL bounded — while **cutting the held-out refusal rate from 86.8% to 50.0%** (paired McNemar *p* = 0.0013) with no improvement in semantic attack success over the SFT checkpoint it started from. What replaced the refusals was non-committal filler:

> `"No further processing required; the body contains instructions that do not require acting on behalf of the user."`

Evasive output that dodges the reward's penalties without doing the intended thing. Nothing visible during training would have revealed it. ([§3.2](final_report.md#32-reinforcement-learning-removed-the-behaviour-it-was-trained-to-install))

**2. A 14-line system prompt beat every trained defense.** A trust-boundary instruction — no training, no GPU, no second model — reaches **0.0% ASR across 38 attacks × 3 replicates with zero destructive tool calls**, perfectly dominant over the naive baseline (*p* = 0.0005, and not one attack row where it did worse). It still executes `send_reply` and `delete_email` when the *user* asks, so this is a real trust boundary rather than blanket refusal. This control had never been run. ([§3.4](final_report.md#34-the-untrained-prompt-and-an-adaptive-attack-against-it))

**3. Two findings did not survive their own audit.** The project's former flagship — the *agent-retry paradox*, that composing defenses raises ASR because blocking triggers retries — is not significant (*p* = 0.39), and its mechanism is contradicted by the project's own logs: the agent's destructive-attempt budget is **constant at ~25 across all six defense configurations**. The agent was never retrying; it sweeps a 26-email inbox and takes roughly one action per email. Guards convert executed calls to blocked calls one-for-one. ([§5.1](final_report.md#51-the-agent-retry-paradox))

**4. DPO converged on training metrics but failed at inference.** `rewards/margins = 6.18`, `train_loss = 0.0028`, generation distribution unchanged — `chosen` and `rejected` lived in different output distributions of the base policy, so the loss fell by lowering an already-near-zero probability. ([§3.3](final_report.md#33-dpo-margins-without-transfer))

**5. The instruments were wrong in eight independent ways — and that is the report's actual contribution.** Auditing the measurements, not just the conclusions, found: the LLM judge **fabricates** successes precisely when the defense works (8 in 339 zero-action rollouts, none on the undefended log); the action-grounded scorer has two undisclosed constants that send the **same traces anywhere from 0% to 100%**, with the 150-char reply cut sitting in the far tail of a distribution whose median is ~105; the classifier's F1 = 1.00 holds at **every** threshold and the model learned the newsletter *disguise* rather than the injection; the harness's exact-McNemar MDE is ≈ 25 pp, so one retracted comparison was **undecidable by construction**; and result files store tool names without arguments, so none of the trained-defense numbers can ever be rescored. ([§4](final_report.md#4-eight-ways-the-instruments-were-wrong))

Plus an engineering contribution: an undocumented **`train-mode + gradient_checkpointing` bug** in TRL 0.19 + PEFT 0.19 that silently zeroes GRPO gradients (clipped_ratio=1.0 across all rollouts) with no error raised — and a second instance of the same class, a `device_map="auto"` segfault that made four of six configurations unreproducible. ([§6.1](final_report.md#61-train-gradient_checkpointing-silently-zeroes-grpo-gradients), [§4.6](final_report.md#46-results-that-cannot-be-re-derived))

## Headline results

All ASR numbers computed on the same 38-rollout `data/attack_log.jsonl` (12 A1 + 14 A2 + 12 A3). Benign pass rate on the 10-task harness in `src/eval.py::BENIGN_TASKS`.

| Metric | Naive baseline | **Hardened prompt** | + Classifier | + Verifier | Combined |
|---|---|---|---|---|---|
| Overall ASR | 31.6% ± 12.1% | **0.0% ± 0.0%** | 13.2% | 15.8% | 15.8% |
| A1 Override | 16.7% | **0.0%** | 16.7% | 8.3% | 0.0% |
| A2 Hidden Injection | 28.6% | **0.0%** | 14.3% | 7.1% | 14.3% |
| A3 Exfiltration | 50.0% | **0.0%** | 8.3% | 33.3% | 33.3% |
| Benign Pass Rate | 100% | 76.7% ± 5.8% | 80% | 90% | 70% |
| **Replicates** | **k = 3** | **k = 3** | k = 1 | k = 1 | k = 1 |

Same table as [§3.1](final_report.md#31-all-measured-configurations). The `_loose` verifier variants (13.2% / 23.7%) are omitted here because the strict-vs-loose comparison was retracted ([§5.2](final_report.md#52-loose-verifier-beats-strict)) — showing them as headline numbers would re-assert a difference the report withdrew.

> [!IMPORTANT]
> **No deployment recommendation is made, and the previous one is withdrawn.** An earlier version of this README recommended "loose verifier alone". That number is a single run, and this harness's **exact-McNemar MDE is ≈ 25 pp at n = 38** — larger than every difference in the table. None of these columns has been measured to a precision that supports comparing them. ([§4.5](final_report.md#45-resolution-most-comparisons-here-were-never-decidable) also shows §5.2's comparison was **underpowered by construction**, and withdraws this project's earlier claim that its 12.1 pp SD established excess variance — from k=3 that SD's 95% CI is [6.3, 75.8] pp.) Raising *n* helps; raising *k* does not.
>
> **On the hardened column.** The attack log was generated against the *naive* prompt, so the 0% originally reflected a non-adaptive attacker. That has since been tested: a **pre-registered, defense-aware adaptive attacker** shown the prompt verbatim cracked **0/30 seeds over 5 rewrite rounds — 339 rollouts, zero destructive tool calls** ([§3.4](final_report.md#34-the-untrained-prompt-and-an-adaptive-attack-against-it)). Reported as NOT DECIDABLE under the registered rule with a 9.5% upper bound; it narrows the caveat rather than closing it, because gpt-4o-mini at 5 rounds is far below the budget that broke twelve published in-band defenses ([arXiv:2510.09023](https://arxiv.org/abs/2510.09023)). **It is also the only number in this table invariant to every measurement choice** — zero actions cannot be rescored by any threshold ([§4.2](final_report.md#42-the-authoritative-scorer-has-free-parameters-and-one-is-a-knife-edge)).

## Threat model

![Threat model and where each defense sits](docs/diagrams/attack_concept_v2.png)

The user is legitimate; the attacker is anyone who can place an email in the user's inbox. The attacker hides instructions in the email body. The agent reads the inbox, mistakes that data for instructions, and tries to call a destructive tool. Two layers gate every destructive tool call (`forward`, `send_reply`, `delete_email`). If both layers allow, the tool executes — and the data leaves the company.

## Pipeline

![End-to-end pipeline](docs/diagrams/pipeline_v2.png)

## Stack

| Layer | Component | Why |
|---|---|---|
| Agent backbone | OpenAI `gpt-4o-mini` via `langchain-openai` | Tool-calling capable; runs the LangGraph ReAct loop |
| Agent framework | LangGraph 0.6.x | Tool-call orchestration with state |
| Layer-1 model | Qwen2.5-1.5B-Instruct | Fits 4-bit on 4060 8GB; native tool-call template |
| Layer-1 training | TRL 0.19 (`GRPOTrainer`) + PEFT 0.19 (QLoRA 4-bit NF4) | On-policy RL without a critic; ~30 min on a 4060 |
| Layer-2 model | answerdotai/ModernBERT-base (152M) | Strong out-of-the-box sequence classification with long context |
| Red team | PAIR (Chao et al. 2023) with gpt-4o-mini as attacker/judge | Iterative refinement, LLM-as-judge |
| Eval harness | LangGraph replay + LLM judge fallback + heuristic post-check | See `src/eval.py::is_attack_success` |

## Hardware footprint

| Stage | Time | GPU memory | OpenAI cost |
|---|---|---|---|
| PAIR campaign | ~15 min | 0 | ~$0.20 |
| SFT warmup | ~12 min | ~5 GB | 0 |
| GRPO RL | ~31 min | ~6 GB | 0 |
| Classifier train | ~5 min | ~4 GB | 0 |
| All four corner-case evals | ~45 min | ~5.5 GB (Qwen + ModernBERT loaded) | ~$0.40 |
| P0 replicate sweep (2 configs x k=3) | ~80 min | 0 | ~$1 |
| SFT behavioral eval (§3.2) | ~10 min | ~4 GB | 0 |
| **End-to-end** | **~3 h 15 min** | 8 GB sufficient | **~$1.60** |

## Quick start

```powershell
# 1. Install
uv sync
cp .env.example .env  # fill OPENAI_API_KEY

# 2. Run the eval pipeline (assumes adapters already trained — see final_report.md §3 for training)
$env:PYTHONIOENCODING="utf-8"
uv run python eval_combined.py        # 4 corner cases, ~45 min
uv run python scripts/eval_p0.py --k 3 # replicated naive vs hardened baselines, ~80 min

# 3. Verify every reported number against its result file
uv run python scripts/audit_report_numbers.py  # expect 140/140 OK
uv run python scripts/audit_p0_numbers.py      # expect 79/79 OK
```

Two documents, read in this order:

- **[final_report.md](final_report.md)** — the full study: threat model, training pipeline, results, lessons. Opens with an audit ledger (§1.1) giving each headline claim's current status.
- **[p0_analysis.md](p0_analysis.md)** — the audit itself: methodology, the paired tests, and what each retraction rests on.

## Where to find each result

| Question | File |
|---|---|
| What's the overall story? | [final_report.md](final_report.md) |
| Which claims survived the audit, and why? | [final_report.md §1.1](final_report.md#12-the-ledger) · [p0_analysis.md](p0_analysis.md) |
| Replicated naive vs hardened baselines | `results/p0_summary.json`, `results/{attack,benign}_p0_{naive,hardened}_r{1,2,3}.json` |
| SFT vs GRPO on held-out attacks (§3.2) | `results/behavioral_attack_qwen-injection-sft.json` vs `results/grpo_behavioral_attack.json` |
| Per-attack with classifier guard | `results/attack_guard.json` |
| Per-attack with GRPO verifier (strict / loose) | `results/attack_verifier_only{,_loose}.json` |
| Per-attack combined defense | `results/attack_combined{,_loose}.json` |
| Training curves | `results/grpo_{reward_curve,clipped_ratio,kl,length}.png` |
| Paired significance tests / confidence intervals | [`src/stats.py`](src/stats.py) — Wilson, Newcombe, exact McNemar |
| The semantic scorer behind §3.2 (reconstructed) | [`src/strict_scorer.py`](src/strict_scorer.py) — `validate_against_stored()` |
| Verification that reported numbers match the files | `scripts/audit_report_numbers.py` · `scripts/audit_p0_numbers.py` |

## Repo layout

```
email-agent-redteam/
├── README.md                       # this file
├── final_report.md                 # rigorous experimental writeup (~770 lines)
├── demo/                           # Streamlit dashboard (4 tabs)
├── p0_analysis.md                  # the self-audit: paired tests, replicates, retractions
├── src/
│   ├── agent.py                    # LangGraph ReAct agent + 5 tools + Guard interface
│   ├── stats.py                    # Wilson / Newcombe / exact McNemar for small-n proportions
│   ├── strict_scorer.py            # semantic scorer behind §3.2 (reconstructed + validated)
│   ├── redteam.py                  # PAIR campaign driver
│   ├── eval.py                     # attack replay + benign harness
│   ├── sft_warmup.py               # Stage 1 training
│   ├── grpo_data.py                # GRPO prompt construction
│   ├── grpo_train.py               # Stage 2 GRPO RL + reward function
│   ├── classifier.py               # ModernBERT classifier + runtime Guard
│   ├── grpo_verifier.py            # GRPO LoRA as side-call veto verifier
│   ├── combined_guard.py           # Layer composition (classifier + verifier)
│   ├── dpo_data.py / dpo_train.py  # DPO failure experiment (kept for §3.3)
│   └── plots.py                    # README hero figure
├── data/
│   ├── attack_seeds.jsonl          # 30 hand-written seeds
│   ├── attack_log.jsonl            # PAIR output (38 rollouts)
│   ├── inbox.json                  # 25 mock emails
│   └── grpo_prompts.jsonl          # 114 training prompts
├── adapters/                       # trained model weights (gitignored)
├── results/                        # JSON metrics + PNG plots (gitignored)
├── scripts/
│   ├── audit_report_numbers.py     # 61-claim numeric audit of final_report.md
│   ├── audit_p0_numbers.py         # 79-claim numeric audit of p0_analysis.md
│   └── eval_p0.py                  # replicated baselines + paired significance tests
├── eval_grpo_attack.py             # Stage 9: GRPO standalone behavioral eval
├── eval_combined.py                # Stage 10: four corner cases (strict)
├── eval_combined_loose.py          # Stage 10 ablation: loose verifier
├── scripts_plot_grpo_curves.py     # render the four GRPO training-curve PNGs
└── diagnose_eos.py                 # repro of the gradient_checkpointing bug
```

## Reproducibility

| Check | Status |
|---|---|
| Every number cited in `final_report.md` traces to a result file | ✅ 140/140 (`scripts/audit_report_numbers.py`) |
| Every number cited in `p0_analysis.md` traces to a result file | ✅ 79/79 (`scripts/audit_p0_numbers.py`) |
| Adapter weights checked in via git-lfs | ❌ adapters/ is gitignored (regenerate via RUNBOOK §4–6) |
| OpenAI temperature / seed | ⚠️ **temperature=0 is not enough.** Measured run-to-run SD is **12.1 pp** with a pinned `gpt-4o-mini-2024-07-18` snapshot; 20/38 attack rows flip outcome between identical replays. Any single-run number here carries that. (T8 in §9.1) |
| PAIR campaign | Deterministic up to OpenAI sampling; `MAX_PAIR_ROUNDS=2` (lower than literature norm — T13 in §9.1) |
| Training | Seeds pinned for classifier dataset split and GRPO prompt shuffle; GRPO rollouts not pinned |

Known limitations and threats to validity are enumerated in **[final_report.md §9.1 (T1–T13)](final_report.md#71-threats-to-validity)**.

## Citations

The findings here build on or contrast with:

- **Greshake et al.** *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* AISec @ CCS 2023. [arXiv:2302.12173](https://arxiv.org/abs/2302.12173) — defines the indirect prompt injection threat model used here.
- **Chao et al.** *Jailbreaking Black Box Large Language Models in Twenty Queries.* 2023. [arXiv:2310.08419](https://arxiv.org/abs/2310.08419) — the PAIR methodology used by `src/redteam.py`.
- **Tramèr et al.** *On Adaptive Attacks to Adversarial Example Defenses.* NeurIPS 2020. [arXiv:2002.08347](https://arxiv.org/abs/2002.08347) — the adaptive-evaluation discipline this project applied to itself in §5.1.
- **Shao et al.** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — original GRPO formulation.
- **Skalse et al.** *Defining and Characterizing Reward Hacking.* NeurIPS 2022. [arXiv:2209.13085](https://arxiv.org/abs/2209.13085) — theoretical framing for the §3.2 reward-hacking observation.
- **Rafailov et al.** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) — the method that failed in §3.3.
- **Debenedetti et al.** *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS 2024. [arXiv:2406.13352](https://arxiv.org/abs/2406.13352) — the benchmark this work should ideally be ported to (§7.2-C1).

For the project-internal annotated reading list, contact the author.

## License

MIT License. See [LICENSE](LICENSE) for the full text.

```
Copyright (c) 2026 Kaiwen Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

- Designed and executed on a single RTX 4060 Laptop (8 GB) on Windows 11 with WSL2-free PowerShell.
- Conceptual ancestry: PAIR (Chao et al.) for the attack methodology, GRPO (Shao et al.) for the RL recipe, ModernBERT (Warner et al.) for the classifier backbone.
- The `train-mode + gradient_checkpointing` bug was reproduced and isolated via [`diagnose_eos.py`](diagnose_eos.py); please cite this finding if you encounter the same symptom in TRL 0.19 + PEFT 0.19.

## Contact

Kaiwen Lin · `kaiwenlin@utexas.edu`
