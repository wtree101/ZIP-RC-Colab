"""Shared helpers for the stage-oriented ZIP-RC notebooks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd


class Gate(TypedDict):
    name: str
    passed: bool
    detail: str
    kind: str


_PROGRESS_ARTIFACT_FLAGS = (
    "--metrics-path",
    "--output-json",
    "--out-parquet",
    "--out",
    "--data",
    "--data-path",
)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository from either its root or the notebooks directory."""
    origin = (start or Path.cwd()).resolve()
    candidates = [
        origin,
        *origin.parents,
        Path("/content/ZIP-RC-Colab"),
        Path("/content/ZIP-RC"),
    ]
    for candidate in candidates:
        if (candidate / "src" / "generate_ziprc_rollouts.py").exists():
            return candidate
    raise FileNotFoundError("Could not find the ZIP-RC repository root.")


def load_config(repo: Path) -> dict[str, object]:
    path = repo / "artifacts" / "experiment_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 00_memory_and_config.ipynb first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def select_best_cuda_device(min_free_gib: float = 4.0) -> str | None:
    """Pin multi-GPU notebook jobs to the visible GPU with the most free memory."""
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current and len([item for item in current.split(",") if item.strip()]) == 1:
        return current

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Could not query GPU memory; leaving CUDA_VISIBLE_DEVICES unchanged: {error}")
        return None

    allowed = None
    if current:
        try:
            allowed = {int(item.strip()) for item in current.split(",") if item.strip()}
        except ValueError:
            print(f"Unsupported CUDA_VISIBLE_DEVICES={current!r}; leaving it unchanged.")
            return current

    devices: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            device_id, free_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if allowed is None or device_id in allowed:
            devices.append((free_mib, device_id))

    if not devices:
        return None
    free_mib, device_id = max(devices)
    if free_mib < min_free_gib * 1024:
        print(
            f"No visible GPU has {min_free_gib:.1f} GiB free; selecting GPU {device_id} "
            f"with {free_mib / 1024:.1f} GiB free."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return str(device_id)


def _command_argument(command: Sequence[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _command_progress_path(repo: Path, command: Sequence[str]) -> Path | None:
    if len(command) < 2 or Path(command[1]).suffix != ".py":
        return None
    script = Path(command[1]).stem
    artifact = next(
        (
            value
            for flag in _PROGRESS_ARTIFACT_FLAGS
            if (value := _command_argument(command, flag)) is not None
        ),
        "run",
    )
    artifact_name = Path(artifact).stem or "run"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{script}__{artifact_name}")
    return repo / "artifacts" / "progress" / f"{safe_name}.json"


def _mark_command_failed(path: Path, error: subprocess.CalledProcessError) -> None:
    payload: dict[str, object] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload["status"] = "failed"
    payload.setdefault("error", f"exit code {error.returncode}")
    payload["exit_code"] = error.returncode
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_repo(repo: Path, *args: object) -> None:
    """Run a command from the repository root and fail on non-zero exit."""
    command = [str(arg) for arg in args]
    python_executable = os.environ.get("ZIPRC_PYTHON")
    if not python_executable:
        local_python = repo / ".venv" / "bin" / "python"
        if local_python.is_file():
            python_executable = str(local_python)
    if python_executable and command and command[0] in {"python", "python3"}:
        command[0] = python_executable
    progress_path = _command_progress_path(repo, command)
    environment = os.environ.copy()
    if progress_path is not None:
        environment["ZIPRC_PROGRESS_PATH"] = str(progress_path)
        print("Progress checkpoint:", progress_path, flush=True)
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=repo, check=True, env=environment)
    except subprocess.CalledProcessError as error:
        if progress_path is not None:
            try:
                _mark_command_failed(progress_path, error)
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
                print(
                    f"Command failed: {payload.get('error', f'exit code {error.returncode}')}",
                    flush=True,
                )
            except OSError as checkpoint_error:
                print(f"Could not mark progress as failed: {checkpoint_error}", flush=True)
        raise
    finally:
        elapsed = time.perf_counter() - started
        print(f"Command elapsed time: {elapsed / 60:.1f} minutes", flush=True)


def _format_seconds(value: object) -> str:
    if not isinstance(value, (int, float)) or value < 0:
        return "—"
    seconds = round(value)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def progress_frame(repo: Path) -> pd.DataFrame:
    """Return the latest durable progress snapshots as a compact display table."""
    directory = repo / "artifacts" / "progress"
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        completed = payload.get("completed")
        total = payload.get("total")
        progress = f"{completed:g}/{total:g}" if isinstance(completed, (int, float)) and isinstance(total, (int, float)) else "—"
        updated_at = payload.get("updated_at")
        age_minutes: float | None = None
        if isinstance(updated_at, str):
            try:
                updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age_minutes = (datetime.now(timezone.utc) - updated).total_seconds() / 60
            except ValueError:
                pass
        rows.append(
            {
                "任务": payload.get("job", path.stem),
                "阶段": payload.get("phase", "—"),
                "状态": payload.get("status", "unknown"),
                "进度": progress,
                "百分比": round(float(payload["percent"]), 1) if isinstance(payload.get("percent"), (int, float)) else None,
                "已耗时": _format_seconds(payload.get("job_elapsed_seconds", payload.get("elapsed_seconds"))),
                "预计剩余": _format_seconds(payload.get("remaining_seconds")),
                "预计完成": payload.get("eta", "—"),
                "距上次更新/分钟": round(age_minutes, 1) if age_minutes is not None else None,
                "文件": path.name,
            }
        )
    return pd.DataFrame(rows)


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
