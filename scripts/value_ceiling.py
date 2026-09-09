"""The linear value ceiling on a pinned record set, with and without the rule inputs.

    uv run --no-project --python .venv/bin/python scripts/value_ceiling.py [checkpoint]

`data/pinned-40k-2026-09-09.pt` is 40 000 uniform draws from `t24h-reinject-lr6`'s
replay window (seed 20260909), frozen to a file the first time this runs so every
checkpoint is scored on the same positions (the 2026-08-21 method rule). It is under
`data/`, which is not tracked; the draw is seeded and the source replay is on disk,
but a regenerated set is the same positions only while that replay is unchanged.

For a checkpoint: the input of its value head (a forward hook on `net.value`), the
two rule inputs of `BrokefishNet.rule_inputs` (each token's legal destinations and
its attacked bit, 2080 columns), three side-to-move summaries (own pieces attacked,
their pieces attacked, material difference), and ridge-OLS held-out correlations with
the game outcome on a 32k/8k split. The question it answers, 2026-09-09: do the rule
inputs carry outcome information the trunk does not already expose linearly? They
did not, on three checkpoints (`docs/journal/2026-09-09-blunder-dissection.md`).
Evaluation only; nothing here reaches a training path.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, ".")
from brokefish.train.buffer import ReplayBuffer
from brokefish.nn.model import BrokefishNet, net_for_state
from brokefish.eval.layer0 import load_net_state
from brokefish.env import cuda_impl as cenv

PIN = "data/pinned-40k-2026-09-09.pt"
if not os.path.exists(PIN):
    buf = ReplayBuffer("runs/t24h-reinject-lr6/replay/t24h-reinject-lr6.dat", window_games=20000,
                       mean_plies=350, resume=True, seed=20260909)
    buf.load("runs/t24h-reinject-lr6/checkpoints/t24h-reinject-lr6.pt.buffer.npz")
    b = buf.sample(40000, device="cpu")
    torch.save({"board": b.board, "control": b.control, "rep": b.rep, "value": b.value,
                "root_value": b.root_value, "weight_gen": b.weight_gen,
                "source": "runs/t24h-reinject-lr6/replay, 40000 uniform draws, seed 20260909, 2026-09-09"}, PIN)
    print("pinned", PIN)
d = torch.load(PIN)
boards, control, rep, z = (d[k].cuda() for k in ("board", "control", "rep", "value"))
N = boards.shape[0]

state = load_net_state(sys.argv[1] if len(sys.argv) > 1 else "runs/t12h-wdl/checkpoints/t12h-wdl.pt", device="cpu")
net = net_for_state(state); net.load_state_dict(state); net = net.cuda().eval()
feats = []
net.value.register_forward_hook(lambda m, i, o: feats.append(i[0].detach().float()))
alive = ((boards.to(torch.int32) >> 11) & 1) == 0
dest_all, att_all = [], []
with torch.no_grad():
    for i in range(0, N, 1024):
        sl = slice(i, i + 1024)
        net(boards[sl], control[sl], rep[sl])
        mask, _ = cenv.movegen(boards[sl], control[sl])
        dest, att = net.rule_inputs(boards[sl], control[sl], alive[sl], mask=mask)
        dest_all.append(dest.reshape(dest.shape[0], -1).float()); att_all.append(att.float())
X_trunk = torch.cat(feats); X_dest = torch.cat(dest_all); X_att = torch.cat(att_all)
# side-to-move summaries: own pieces attacked, their pieces attacked, material difference
stm_black = control < 0
colour = ((boards.to(torch.int32) >> 10) & 1).bool()          # 1 = black piece
own = (colour == stm_black[:, None]) & alive
VAL = torch.tensor([1, 3, 3, 5, 9, 0], device="cuda", dtype=torch.float)
typ = (boards.to(torch.int32) >> 6) & 7
mat = VAL[typ.clamp(max=5)] * alive
X_sum = torch.stack([(X_att * own).sum(1), (X_att * (~own & alive)).sum(1),
                     (mat * own).sum(1) - (mat * (~own)).sum(1)], 1)

def ceiling(X, name, lam=1e-2):
    X = X.double(); y = z.double()
    X = torch.cat([X, torch.ones(N, 1, device="cuda", dtype=torch.double)], 1)
    tr, te = slice(0, 32000), slice(32000, N)
    A = X[tr].T @ X[tr] + lam * X.shape[1] * torch.eye(X.shape[1], device="cuda", dtype=torch.double)
    w = torch.linalg.solve(A, X[tr].T @ y[tr])
    p = X[te] @ w
    c = torch.corrcoef(torch.stack([p, y[te]]))[0, 1].item()
    ctr = torch.corrcoef(torch.stack([X[tr] @ w, y[tr]]))[0, 1].item()
    print(f"{name:34s} d={X.shape[1]-1:5d}  held-out corr {c:.4f}  (train {ctr:.4f})")
print("N", N, "decisive", float((z != 0).float().mean()))
ceiling(X_sum[:, 2:3], "material only")
ceiling(X_sum, "summaries: att own/their, material")
ceiling(X_att, "attacked bits, 32")
ceiling(torch.cat([X_dest, X_att], 1), "rule inputs, 2080")
ceiling(X_trunk, "trunk (value-head input)")
ceiling(torch.cat([X_trunk, X_att], 1), "trunk + attacked bits")
ceiling(torch.cat([X_trunk, X_sum], 1), "trunk + summaries")
ceiling(torch.cat([X_trunk, X_dest, X_att], 1), "trunk + rule inputs")
