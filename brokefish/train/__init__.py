"""The training loop, ``docs/train.md``.

C2: play games with the current weights, store what the search produced, sample from
the store, take gradient steps, publish new weights, repeat. Six modules, and the
split follows the document rather than convenience.

``loss``    §3, AZ eq. (1), and the label decode it turns on. An independent
            transcription of ``_expand``'s enumeration, because a mismatch permutes
            the target silently and the loss falls the whole time it happens.
``buffer``  §5, the 500,000-game window as a memory-mapped ring, eviction by game,
            and the §4 parity rule that turns one game result into a value per ply.
``sync``    §8.1, the two weight representations and the assertion that stops a run
            from self-playing generation-0 weights forever.
``log``     §11, JSONL plus wandb when it is installed.
``loop``    §§6-10, the alternation, the cadence, the optimiser, the checkpoint and
            the euro counter.   ``python -m brokefish.train.loop``
``overfit`` §12 check 1, which is the first thing to run.
            ``python -m brokefish.train.overfit``
"""

from .buffer import RECORD, RECORD_BYTES, ReplayBuffer
from .log import Logger
from .loss import (TrainBatch, audit_labels, az_loss, edge_logits, l2_penalty,
                   weight_decay_for)
from .sync import PackedWeights, weight_fingerprint
from .loop import LR_SCHEDULE, TrainConfig, Trainer

__all__ = [
    "RECORD", "RECORD_BYTES", "ReplayBuffer",
    "TrainBatch", "az_loss", "audit_labels", "edge_logits", "l2_penalty", "weight_decay_for",
    "TrainConfig", "Trainer", "LR_SCHEDULE",
    "PackedWeights", "weight_fingerprint", "Logger",
]
