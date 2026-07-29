"""Download one shard of human games, for testing the move generator only.

The shard feeds `brokefish.env.positions`, which draws random positions for the
differential tests against python-chess. It is not committed and never reaches
training: see the boundary note at the top of that module.

Default is a random shard of the most recent month on the Lichess parquet mirror,
so the test set rolls forward on its own rather than being pinned to one snapshot
of one era of play. Standard library only, no huggingface_hub.

⚠️ Recent months are sharded at about 1 GB per file, against 37 MB for the whole
of 2013-01. The size is printed before the download starts. To pin the small one:

    python scripts/fetch_test_positions.py                       # latest month, random shard
    python scripts/fetch_test_positions.py --year 2013 --month 01  # 37 MB
"""

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

REPO = "Lichess/standard-chess-games"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def tree(path: str):
    with urllib.request.urlopen(f"{API}/{path}", timeout=60) as r:
        return json.load(r)


def latest_month() -> tuple:
    """The most recent (year, month) the mirror publishes."""
    year = sorted(e["path"] for e in tree("data") if e["type"] == "directory")[-1]
    month = sorted(e["path"] for e in tree(year) if e["type"] == "directory")[-1]
    return year.split("=")[-1], month.split("=")[-1]


def list_shards(year: str, month: str):
    entries = tree(f"data/year={year}/month={month}")
    return sorted((e for e in entries if e["type"] == "file" and e["path"].endswith(".parquet")),
                  key=lambda e: e["path"])


def download(path: str, size: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"{RESOLVE}/{path}  ({size / 1e6:.0f} MB)\n  -> {dest}")
    with urllib.request.urlopen(f"{RESOLVE}/{path}", timeout=600) as r, open(dest, "wb") as f:
        total = int(r.headers.get("content-length", size))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", help="default: the most recent published")
    p.add_argument("--month", help="default: the most recent published")
    p.add_argument("--shard", type=int, help="shard index; default: random")
    p.add_argument("--out", type=Path, default=DATA_DIR)
    p.add_argument("--force", action="store_true", help="re-download if present")
    args = p.parse_args()

    if args.year and args.month:
        year, month = args.year, args.month
    else:
        year, month = latest_month()
        print(f"latest published month: {year}-{month}")

    shards = list_shards(year, month)
    if not shards:
        print(f"no shard for {year}-{month}", file=sys.stderr)
        return 1
    shard = shards[args.shard] if args.shard is not None else random.choice(shards)

    dest = args.out / f"{year}-{month}-{Path(shard['path']).name}"
    if dest.exists() and not args.force:
        print(f"{dest} already there ({dest.stat().st_size / 1e6:.0f} MB); --force to replace")
        return 0
    download(shard["path"], shard["size"], dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
