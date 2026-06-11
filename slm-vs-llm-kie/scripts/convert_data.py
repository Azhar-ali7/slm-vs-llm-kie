"""Convert datasets YOU have already downloaded into the framework's JSONL.

This does NOT download anything. Run `python scripts/download_data.py` first to
see the exact manual download commands, place the files, then run this.

Expected layout after your manual download (defaults; override with flags):

  SROIE  (cloned repo zzzDavid/ICDAR-2019-SROIE):
    data/raw/sroie/_src/data/box/*.txt
    data/raw/sroie/_src/data/key/*.txt
      -> writes data/raw/sroie/{train,test}.jsonl  (seeded 80/20 split)

  Kleister-Charity (raw blobs from applicaai/kleister-charity):
    data/raw/kleister_charity/_src/train/in.tsv.xz   + expected.tsv
    data/raw/kleister_charity/_src/dev-0/in.tsv.xz   + expected.tsv
      -> writes data/raw/kleister_charity/{train,test}.jsonl   (dev-0 -> test)

Usage:
  python scripts/convert_data.py                 # auto-detect _src/ folders
  python scripts/convert_data.py --sroie-repo /path/to/ICDAR-2019-SROIE
  python scripts/convert_data.py --kleister-dir /path/to/kleister-charity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.convert import (  # noqa: E402
    kleister_records_from_tsv,
    seeded_split,
    sroie_records_from_dirs,
    write_jsonl,
)


def _find_sroie_box_key(repo: Path) -> tuple[Path, Path] | None:
    for box in repo.rglob("box"):
        key = box.parent / "key"
        if box.is_dir() and key.is_dir():
            return box, key
    return None


def convert_sroie(cfg, repo: Path) -> None:
    raw_dir = resolve_path(cfg, cfg["datasets"]["sroie"]["raw_dir"])
    found = _find_sroie_box_key(repo)
    if not found:
        print(f"[sroie] no box/ + key/ folders under {repo} — skipping.")
        return
    box_dir, key_dir = found
    records = sroie_records_from_dirs(box_dir, key_dir, split="all")
    if not records:
        print(f"[sroie] found folders but no labelled records in {box_dir} — skipping.")
        return
    train, test = seeded_split(records, test_frac=0.2, seed=cfg.get("seed", 42))
    n_tr = write_jsonl(train, raw_dir / "train.jsonl")
    n_te = write_jsonl(test, raw_dir / "test.jsonl")
    print(f"[sroie] {len(records)} labelled receipts -> train={n_tr}, test={n_te}  "
          f"({raw_dir})")


def convert_kleister(cfg, src: Path) -> None:
    raw_dir = resolve_path(cfg, cfg["datasets"]["kleister_charity"]["raw_dir"])
    head = 60
    max_lines = 250
    pairs = [("train", "train", "train.jsonl"), ("dev-0", "test", "test.jsonl")]
    wrote_any = False
    for sub, split, out_name in pairs:
        in_xz = src / sub / "in.tsv.xz"
        exp = src / sub / "expected.tsv"
        if not in_xz.exists():
            print(f"[kleister] missing {in_xz} — skipping {sub}.")
            continue
        records = kleister_records_from_tsv(in_xz, exp, split=split,
                                            head=head, max_lines=max_lines)
        n = write_jsonl(records, raw_dir / out_name)
        gold = "with gold" if exp.exists() else "NO gold (test-A withheld)"
        print(f"[kleister] {sub} -> {out_name}: {n} docs ({gold})")
        wrote_any = True
    if not wrote_any:
        print(f"[kleister] nothing converted — expected train/ and dev-0/ under {src}.")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Convert downloaded datasets to JSONL.")
    ap.add_argument("--sroie-repo", default=None,
                    help="Path to the cloned ICDAR-2019-SROIE repo (contains data/box, data/key).")
    ap.add_argument("--kleister-dir", default=None,
                    help="Path containing train/ and dev-0/ (in.tsv.xz + expected.tsv).")
    args = ap.parse_args()

    sroie_repo = Path(args.sroie_repo) if args.sroie_repo else \
        resolve_path(cfg, cfg["datasets"]["sroie"]["raw_dir"]) / "_src"
    kleister_dir = Path(args.kleister_dir) if args.kleister_dir else \
        resolve_path(cfg, cfg["datasets"]["kleister_charity"]["raw_dir"]) / "_src"

    if sroie_repo.exists():
        convert_sroie(cfg, sroie_repo)
    else:
        print(f"[sroie] {sroie_repo} not found — download SROIE first "
              f"(see: python scripts/download_data.py).")

    if kleister_dir.exists():
        convert_kleister(cfg, kleister_dir)
    else:
        print(f"[kleister] {kleister_dir} not found — download Kleister first "
              f"(see: python scripts/download_data.py).")

    print("\nNext: rm -f results/test_set_20.json  (rebuild the 20-doc set on real data), "
          "then python scripts/run_eval.py --pilot")


if __name__ == "__main__":
    main()
