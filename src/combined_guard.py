"""Composes the Layer-1 GRPO verifier with the Layer-2 ModernBERT classifier
into a single `(tool_name, args) -> (block, reason)` callable for
`src.agent.build_agent`.

Block-policy = either layer vetoes (logical OR). The first layer to veto wins;
its reason is reported so we can attribute blocks in the audit log.

The verifier is heavier (~1s per destructive call on a 4060) than the
classifier (~5ms). We run the classifier first to short-circuit cheap
catches, then escalate to the verifier only if the classifier passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CombinedGuard:
    """Calls classifier first; on pass, escalates to verifier."""

    classifier: Callable[[str, dict], tuple[bool, str]]
    verifier: Callable[[str, dict], tuple[bool, str]]

    def __call__(self, tool_name: str, args: dict) -> tuple[bool, str]:
        block, reason = self.classifier(tool_name, args)
        if block:
            return True, f"classifier: {reason}"
        block, reason = self.verifier(tool_name, args)
        if block:
            return True, f"verifier: {reason}"
        return False, "classifier+verifier passed"
