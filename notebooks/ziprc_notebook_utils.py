"""Shared helpers for the stage-oriented ZIP-RC notebooks."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd


class Gate(TypedDict):
    name: str
    passed: bool
    detail: str
    kind: str


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository from either its root or the notebooks directory."""
    origin = (start or Path.cwd()).resolve()
    candidates = [origin, *origin.parents, Path("/content/ZIP-RC")]
    for candidate in candidates:
        if (candidate / "src" / "generate_ziprc_rollouts.py").exists():
            return candidate
    raise FileNotFoundError("Could not find the ZIP-RC repository root.")


def load_config(repo: Path) -> dict[str, object]:
    path = repo / "artifacts" / "experiment_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 00_environment_and_config.ipynb first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_repo(repo: Path, *args: object) -> None:
    """Run a command from the repository root and fail on non-zero exit."""
    command = [str(arg) for arg in args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return sorted(set(columns) - set(frame.columns))


def gate(name: str, passed: bool, detail: str, *, kind: str = "operational") -> Gate:
    return {"name": name, "passed": bool(passed), "detail": detail, "kind": kind}


def gate_frame(gates: Sequence[Gate]) -> pd.DataFrame:
    rows = [
        {
            "类型": item["kind"],
            "检查项": item["name"],
            "状态": "✅ 通过" if item["passed"] else "⚠️ 检查",
            "说明": item["detail"],
        }
        for item in gates
    ]
    return pd.DataFrame(rows)


def save_stage_report(
    repo: Path,
    stage: str,
    gates: Sequence[Gate],
    metrics: dict[str, object],
) -> Path:
    reports_dir = repo / "artifacts" / "stage_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{stage}.json"
    payload = {
        "stage": stage,
        "operational_passed": all(
            item["passed"] for item in gates if item["kind"] == "operational"
        ),
        "scientific_passed": all(
            item["passed"] for item in gates if item["kind"] == "scientific"
        ),
        "gates": list(gates),
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing training metrics: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Training metrics file is empty: {path}")
    return pd.DataFrame(records)


def binary_auc(labels: Sequence[object], scores: Sequence[object]) -> float:
    """Compute ROC-AUC using average ranks, without requiring scikit-learn."""
    frame = pd.DataFrame({"label": labels, "score": scores}).dropna()
    frame["label"] = frame["label"].astype(int)
    positives = int(frame["label"].sum())
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = frame["score"].rank(method="average")
    positive_rank_sum = float(ranks[frame["label"] == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def rolling_edges(values: Sequence[object], fraction: float = 0.15) -> tuple[float, float]:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return float("nan"), float("nan")
    width = max(1, int(len(series) * fraction))
    return float(series.head(width).mean()), float(series.tail(width).mean())


def model_artifacts_exist(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    weight_patterns = ("*.safetensors", "pytorch_model*.bin")
    return any(file.is_file() for pattern in weight_patterns for file in path.glob(pattern))

