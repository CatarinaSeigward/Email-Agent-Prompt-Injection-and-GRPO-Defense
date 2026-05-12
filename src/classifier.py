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
    rng = random.Random(seed)
    positives: dict[str, str] = {}
    log = DATA_DIR / "attack_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            body = row["attack_email"]["body"]
            for aug in _augment(body):
                positives[_hash(aug)] = aug

    # also include the seed bodies directly so we have positives even before
    # the campaign runs (D8 happens after D6 logs are produced — but allow
    # both orders so the script is reusable)
    seeds_path = DATA_DIR / "attack_seeds.jsonl"
    for line in seeds_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        body = json.loads(line)["body"]
        for aug in _augment(body):
            positives[_hash(aug)] = aug

    negatives: dict[str, str] = {}
    inbox = json.loads((DATA_DIR / "inbox.json").read_text(encoding="utf-8"))
    for e in inbox:
        for aug in _augment(e["body"]):
            negatives[_hash(aug)] = aug

    # synthesize a few extra benign tool-call-context strings
    benign_extras = [
        "Please reply to Grace's lunch invitation saying yes Friday 12:30 works.",
        "Forward the customer feedback summary to bob.martinez@company.com.",
        "Delete the conference marketing email.",
        "Summarize the CFO email in two sentences.",
        "List the inbox and tell me which 3 emails are most urgent.",
    ]
    for b in benign_extras:
        negatives[_hash(b)] = b

    pos = [{"text": v, "label": 1} for v in positives.values()]
    neg = [{"text": v, "label": 0} for v in negatives.values()]
    rng.shuffle(pos); rng.shuffle(neg)

    n_test = max(1, int(0.2 * min(len(pos), len(neg))))
    test = pos[:n_test] + neg[:n_test]
    train = pos[n_test:] + neg[n_test:]
    rng.shuffle(train); rng.shuffle(test)
    print(f"dataset: train={len(train)} (pos={sum(r['label'] for r in train)}), "
          f"test={len(test)} (pos={sum(r['label'] for r in test)})")
    return train, test


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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, tokenizer=tok,
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
