"""Instrumentation, ``docs/train.md`` §11.

Two sinks, and the file one is not optional. Every record goes to a JSONL file that
survives the process, and to wandb — **online by default**. A human-readable ``.log``
beside it is what ``tail -f`` is pointed at.

Online is safe: wandb writes every record to ``wandb/run-*/`` on disk first and uploads
from a background thread that retries, so a dropped network stalls the sync, it does not
stall or kill the run. ``--wandb-mode offline`` is for a machine with no credentials (a
fresh rented GPU); push it afterwards with ``python -m wandb sync wandb/offline-run-*``.

⚠️ **wandb is a hard dependency of this module as of 2026-07-31** — it is installed in
``.venv`` and imported at the top like anything else. It is **not yet in
``requirements.lock``**, so a fresh machine (rented GPU, CI) needs::

    .venv/bin/pip install wandb && .venv/bin/pip freeze > requirements.lock

until that lands, or ``import brokefish.train`` fails there while working here.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import wandb


def _flatten(prefix: str, obj) -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(f"{prefix}/{k}" if prefix else str(k), v))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten(f"{prefix}/{i}", v))
    else:
        out[prefix] = obj
    return out


class Logger:
    """JSONL plus an optional wandb run. Nothing here ever raises on a logging path."""

    def __init__(self, run: str, log_dir: str = "logs", config: Optional[dict] = None,
                 use_wandb: bool = True, wandb_mode: str = "online",
                 project: str = "brokefish", append: bool = False) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.run = run
        self.jsonl_path = os.path.join(log_dir, f"{run}.jsonl")
        self.text_path = os.path.join(log_dir, f"{run}.log")
        mode = "a" if append else "w"
        self._jsonl = open(self.jsonl_path, mode, buffering=1)
        self._text = open(self.text_path, mode, buffering=1)
        self.t0 = time.time()

        self.wandb = None
        self.wandb_why = "disabled"
        if use_wandb:
            # `init` still fails softly: a run that dies because a dashboard would
            # not start is an expensive way to discover a network problem, and the
            # JSONL above is the sink that matters. The *import* is unconditional;
            # the *session* is not.
            try:
                self.wandb = wandb.init(project=project, name=run, config=config or {},
                                        mode=wandb_mode, reinit=True)
                self.wandb_why = f"wandb {wandb.__version__}, mode={wandb_mode}"
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                self.wandb = None
                self.wandb_why = f"wandb init failed, JSONL only: {type(exc).__name__}: {exc}"

    # -- sinks -------------------------------------------------------------- #

    def note(self, line: str = "") -> None:
        """A human line. Goes to stdout and the ``.log``, never to wandb."""
        print(line, flush=True)
        self._text.write(line + "\n")

    def log(self, data: dict, step: Optional[int] = None) -> None:
        flat = _flatten("", data)
        flat["wall_s"] = time.time() - self.t0
        if step is not None:
            flat["step"] = step
        self._jsonl.write(json.dumps(flat, default=float) + "\n")
        if self.wandb is not None:
            self.wandb.log(flat, step=step)

    def summary(self, data: dict) -> None:
        self.log({"summary": data})
        if self.wandb is not None:
            for k, v in _flatten("", data).items():
                self.wandb.summary[k] = v

    def close(self) -> None:
        self._jsonl.close()
        self._text.close()
        if self.wandb is not None:
            self.wandb.finish()
