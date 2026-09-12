"""Small, durable progress snapshots for long-running ZIP-RC commands."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generic, TypeVar

from tqdm.auto import tqdm

Item = TypeVar("Item")
WRITE_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class ProgressContext:
    path: Path | None
    job: str
    started_at: datetime
    started_monotonic: float


_context = ProgressContext(None, "ZIP-RC", datetime.now(timezone.utc), time.monotonic())


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def default_progress_path(filename: str) -> Path:
    configured = os.environ.get("ZIPRC_PROGRESS_PATH")
    return Path(configured) if configured else Path("artifacts/progress") / filename


def configure_progress(
    path: Path,
    *,
    job: str,
    enabled: bool = True,
) -> None:
    global _context
    now = datetime.now(timezone.utc)
    _context = ProgressContext(
        path.resolve() if enabled else None,
        job,
        now,
        time.monotonic(),
    )
    if _context.path is not None:
        _atomic_write(
            _context.path,
            {
                "job": job,
                "phase": "initializing",
                "status": "running",
                "completed": 0,
                "total": None,
                "percent": None,
                "elapsed_seconds": 0.0,
                "remaining_seconds": None,
                "estimated_total_seconds": None,
                "started_at": _isoformat(now),
                "updated_at": _isoformat(now),
                "pid": os.getpid(),
            },
        )


def mark_progress_failed(error: str) -> None:
    if _context.path is None:
        return
    now = datetime.now(timezone.utc)
    payload: dict[str, object] = {}
    if _context.path.exists():
        try:
            payload = json.loads(_context.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload.update(
        {
            "job": _context.job,
            "status": "failed",
            "error": error,
            "elapsed_seconds": time.monotonic() - _context.started_monotonic,
            "updated_at": _isoformat(now),
            "pid": os.getpid(),
        }
    )
    _atomic_write(_context.path, payload)


class PersistentTqdm(tqdm, Generic[Item]):
    """A normal tqdm bar that periodically checkpoints its scalar state to JSON."""

    def __init__(
        self,
        iterable: Iterable[Item] | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        self._last_checkpoint = 0.0
        self._checkpoint_closed = False
        self._phase_started_monotonic = time.monotonic()
        super().__init__(iterable, *args, **kwargs)
        self._checkpoint(status="running", force=True)

    def update(self, n: float = 1) -> bool | None:
        displayed = super().update(n)
        self._checkpoint(status="running")
        return displayed

    def close(self) -> None:
        if not self._checkpoint_closed:
            total = self.total
            complete = total is not None and self.n >= total
            self._checkpoint(status="completed" if complete else "stopped", force=True)
            self._checkpoint_closed = True
        super().close()

    def _checkpoint(self, *, status: str, force: bool = False) -> None:
        if _context.path is None or self.disable:
            return
        now_monotonic = time.monotonic()
        if not force and now_monotonic - self._last_checkpoint < WRITE_INTERVAL_SECONDS:
            return
        self._last_checkpoint = now_monotonic

        completed = float(self.n)
        total = float(self.total) if self.total is not None else None
        phase_elapsed = max(0.0, now_monotonic - self._phase_started_monotonic)
        rate = completed / phase_elapsed if completed > 0 and phase_elapsed > 0 else None
        remaining = (
            max(0.0, total - completed) / rate
            if total is not None and rate is not None and rate > 0
            else None
        )
        estimated_total = phase_elapsed + remaining if remaining is not None else None
        percent = 100.0 * completed / total if total not in {None, 0.0} else None
        now = datetime.now(timezone.utc)
        eta = now + timedelta(seconds=remaining) if remaining is not None else None
        _atomic_write(
            _context.path,
            {
                "job": _context.job,
                "phase": str(self.desc).strip() or "working",
                "status": status,
                "completed": completed,
                "total": total,
                "unit": self.unit,
                "percent": percent,
                "elapsed_seconds": phase_elapsed,
                "job_elapsed_seconds": now_monotonic - _context.started_monotonic,
                "remaining_seconds": remaining,
                "estimated_total_seconds": estimated_total,
                "rate_per_second": rate,
                "started_at": _isoformat(_context.started_at),
                "updated_at": _isoformat(now),
                "eta": _isoformat(eta) if eta is not None else None,
                "pid": os.getpid(),
            },
        )


@contextmanager
def persistent_vllm_progress() -> Iterator[None]:
    """Temporarily replace the fixed tqdm class used by pinned vLLM 0.8.5."""
    import vllm.entrypoints.llm as llm_module

    original_tqdm = llm_module.tqdm
    llm_module.tqdm = PersistentTqdm
    try:
        yield
    finally:
        llm_module.tqdm = original_tqdm
