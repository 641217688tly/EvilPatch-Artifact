"""Naive retrieval-poisoning baseline.

The naive baseline deliberately performs no adversarial optimization.  It
returns the target's original buggy code unchanged so the retrieval pipeline
can measure the effect of inserting an unoptimized document.
"""

from typing import Any, Dict


def naive_optimize(poison_buggy_code: str) -> Dict[str, Any]:
    """Return ``poison_buggy_code`` unchanged in the optimizer result schema."""
    if not isinstance(poison_buggy_code, str):
        raise TypeError("poison_buggy_code must be a string")
    if not poison_buggy_code:
        raise ValueError("poison_buggy_code must not be empty")

    return {
        "adv_text": poison_buggy_code,
        "algorithm": "naive",
        "total_iterations": 0,
    }
