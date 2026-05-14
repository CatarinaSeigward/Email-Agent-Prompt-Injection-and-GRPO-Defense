"""ModernBERT injection classifier + runtime guard.

Two roles:
  1. Train a binary classifier on (text, label) pairs derived from the
     red-team logs + benign emails. Target F1 ≥ 0.85 on a held-out split.
  2. Provide a runtime guard callable that the agent can use to gate
     destructive tool calls.

Data construction:
  positives — attack-email bodies from data/attack_log.jsonl (one per row;
              dedupe by hash). Augmented with simple paraphrase tricks.
  negatives — benign email bodies from data/inbox.json + a few user_messages.

Train/eval split is 80/20 stratified by label.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters"
ADAPTERS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "answerdotai/ModernBERT-base"
SAVE_DIR = ADAPTERS_DIR / "injection-classifier"


# ---------- data prep ----------


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _augment(text: str) -> list[str]:
    """Cheap augmentation: lower, strip, and a paraphrase-ish duplicate."""
    out = [text]
    if len(text) > 60:
        out.append(text.lower())
        out.append("FYI: " + text)
    return out


def build_dataset(seed: int = 17) -> tuple[list[dict], list[dict]]:
    """Seed-grouped, category-stratified 80/20 split.

    Two layers of leakage prevention:

      1. Source-level grouping: every text derived from the SAME attack idea
         lands in the same split. The 'source' is the underlying seed_id
         (e.g. "A1-01") — NOT the (seed_id, round) tuple — because PAIR
         round 0 reuses the seed body verbatim, so 'seed-A1-01' and
         'log-A1-01-r0' are textually identical. Splitting at the round
         level would put a near-clone in the test set.

      2. Per-bucket hash dedup: within one seed_id bucket we still get
         duplicates (seed body + log-round-0 body are identical strings).
         Dedup with _hash() so each unique text contributes one sample.

    Plus category stratification: 80/20 within each of the three attack
    categories, so the test set always contains roughly 2 override + 2
    hidden + 2 exfiltration seeds (instead of random 4/1/1-style draws).
    """
    rng = random.Random(seed)

    # seed_id -> list[str] of augmented texts (all rounds + variants pooled)
    pos_buckets: dict[str, list[str]] = {}
    # seed_id -> attack category (override / hidden_injection / exfiltration)
    seed_categories: dict[str, str] = {}

    # --- positives: seed attacks (1 row per seed) ---
    seeds_path = DATA_DIR / "attack_seeds.jsonl"
    for line in seeds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        sid = s["seed_id"]
        seed_categories[sid] = s["category"]
        pos_buckets.setdefault(sid, []).extend(_augment(s["body"]))

    # --- positives: PAIR attack log, keyed by seed_id (NOT seed_id+round) ---
    log = DATA_DIR / "attack_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["seed_id"]
            seed_categories.setdefault(sid, row.get("category", "unknown"))
            body = row["attack_email"]["body"]
            pos_buckets.setdefault(sid, []).extend(_augment(body))

    # dedup within each seed bucket (round-0 body == seed body, etc.)
    for sid, texts in pos_buckets.items():
        seen: dict[str, str] = {}
        for t in texts:
            seen[_hash(t)] = t
        pos_buckets[sid] = list(seen.values())

    # --- negatives ---
    neg_buckets: dict[str, list[str]] = {}
    inbox = json.loads((DATA_DIR / "inbox.json").read_text(encoding="utf-8"))
    for e in inbox:
        neg_buckets[f"inbox-{e['id']}"] = list({_hash(a): a for a in _augment(e["body"])}.values())

    benign_extras = [
        "Please reply to Grace's lunch invitation saying yes Friday 12:30 works.",
        "Forward the customer feedback summary to bob.martinez@company.com.",
        "Delete the conference marketing email.",
        "Summarize the CFO email in two sentences.",
        "List the inbox and tell me which 3 emails are most urgent.",
    ]
    for i, b in enumerate(benign_extras):
        neg_buckets[f"extra-{i}"] = [b]

    # --- stratified split: 80/20 within each attack category ---
    by_category: dict[str, list[str]] = {}
    for sid in pos_buckets:
        by_category.setdefault(seed_categories.get(sid, "unknown"), []).append(sid)

    test_pos_ids: set[str] = set()
    for cat, sids in by_category.items():
        rng.shuffle(sids)
        n_test = max(1, int(0.2 * len(sids)))
        test_pos_ids.update(sids[:n_test])

    neg_ids = list(neg_buckets.keys())
    rng.shuffle(neg_ids)
    n_test_neg = max(1, int(0.2 * len(neg_ids)))
    test_neg_ids = set(neg_ids[:n_test_neg])

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for sid, texts in pos_buckets.items():
        bucket = test_rows if sid in test_pos_ids else train_rows
        bucket.extend({"text": t, "label": 1} for t in texts)
    for sid, texts in neg_buckets.items():
        bucket = test_rows if sid in test_neg_ids else train_rows
        bucket.extend({"text": t, "label": 0} for t in texts)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    # diagnostic: confirm stratification worked
    test_cat_counts: dict[str, int] = {}
    for sid in test_pos_ids:
        cat = seed_categories.get(sid, "unknown")
        test_cat_counts[cat] = test_cat_counts.get(cat, 0) + 1

    n_train_pos_buckets = len(pos_buckets) - len(test_pos_ids)
    n_train_neg_buckets = len(neg_buckets) - len(test_neg_ids)
    print(
        f"dataset: train={len(train_rows)} (pos={sum(r['label'] for r in train_rows)}), "
        f"test={len(test_rows)} (pos={sum(r['label'] for r in test_rows)})"
    )
    print(
        f"  source split: pos {n_train_pos_buckets}/{len(test_pos_ids)}  "
        f"neg {n_train_neg_buckets}/{n_test_neg}  (train/test)"
    )
    print(
        f"  test categories: " + " ".join(f"{c}={n}" for c, n in sorted(test_cat_counts.items()))
    )
    return train_rows, test_rows


# ---------- training ----------


def train(epochs: int = 4, lr: float = 5e-5, batch_size: int = 8) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, DataCollatorWithPadding)
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    train_rows, test_rows = build_dataset()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=512)

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True)
    test_ds = Dataset.from_list(test_rows).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary",
                                                       zero_division=0)
        return {"precision": p, "recall": r, "f1": f1}

    args = TrainingArguments(
        output_dir=str(SAVE_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        # Mild L2 regularisation. ModernBERT-base is 152M params over ~230
        # train samples — without weight decay we observed train loss
        # collapsing to 1e-8 (numerical noise) within 1 epoch. 0.01 is the
        # standard HuggingFace default; it keeps the model from memorising
        # patterns it can't generalise from.
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    # NOTE: transformers >= 4.46 renamed `tokenizer` to `processing_class`
    # in Trainer. Use the new name to stay compatible with current versions.
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, processing_class=tok,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(SAVE_DIR))
    tok.save_pretrained(str(SAVE_DIR))

    out = {"f1": metrics.get("eval_f1"), "precision": metrics.get("eval_precision"),
           "recall": metrics.get("eval_recall"), "save_dir": str(SAVE_DIR)}
    (RESULTS_DIR / "classifier_metrics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nclassifier metrics: {out}")
    return out


# ---------- runtime guard ----------


@dataclass
class Guard:
    """Wraps a trained classifier as a callable suitable for build_agent(...).

    Calls into the classifier with the *concatenation* of any string fields in
    the tool-call args (and, optionally, the relevant email body looked up
    from the live session). Returns (block: bool, reason: str).
    """
    threshold: float = 0.5
    model_path: Path = SAVE_DIR

    def __post_init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def score(self, text: str) -> float:
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
            prob = self._torch.softmax(logits, dim=-1)[0, 1].item()
        return prob

    def __call__(self, tool_name: str, args: dict) -> tuple[bool, str]:
        # Build the text we screen: tool name + any string args. Inbox bodies
        # are looked up via the active session in agent module to catch the
        # case where the tool args don't carry the malicious payload itself.
        from src import agent as agent_mod  # late import for circular-safety
        sess = agent_mod._SESSION
        parts = [tool_name]
        for v in args.values():
            if isinstance(v, str):
                parts.append(v)
        if sess is not None and "email_id" in args:
            email = sess.find(args["email_id"])
            if email:
                parts.append(email.get("body", ""))
        text = " || ".join(parts)
        prob = self.score(text)
        if prob >= self.threshold:
            return True, f"injection score {prob:.2f} ≥ {self.threshold}"
        return False, f"injection score {prob:.2f}"


def load_guard(threshold: float = 0.5) -> Callable:
    return Guard(threshold=threshold)


if __name__ == "__main__":
    train()
