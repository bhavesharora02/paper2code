# core/verify/metrics.py
"""Small collection of metric helpers used by metric_runner.
Keep this minimal — the metric_runner will look up `expected_metrics['key']` and call
one of these functions.
"""
from typing import Sequence, Any, Dict

def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    preds = list(preds)
    labels = list(labels)
    if not labels:
        return 0.0
    correct = sum(1 for p, y in zip(preds, labels) if int(p) == int(y))
    return correct / len(labels)

# small registry so callers can look up by name
METRIC_REGISTRY: Dict[str, Any] = {
    "accuracy": accuracy,
}