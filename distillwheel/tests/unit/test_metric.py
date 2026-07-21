import json
from pathlib import Path

from distillwheel.core.ir.metric import Metric


def test_metric_json_is_safe_for_non_standard_backend_values():
    recursive = {}
    recursive["self"] = recursive
    metric = Metric(
        step=1,
        timestamp=1.0,
        stage="grpo",
        loss=float("nan"),
        grad_norm=float("inf"),
        extra={
            "set": {2, 1},
            "bytes": b"bad:\xff",
            "path": Path("checkpoint"),
            "nested": {"reward": float("-inf")},
            "recursive": recursive,
        },
    )

    payload = metric.to_json()
    parsed = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert parsed["loss"] is None
    assert parsed["grad_norm"] is None
    assert parsed["extra"]["set"] == [1, 2]
    assert parsed["extra"]["bytes"] == "bad:�"
    assert parsed["extra"]["path"] == "checkpoint"
    assert parsed["extra"]["nested"]["reward"] is None
    assert parsed["extra"]["recursive"]["self"] == "<recursive>"
