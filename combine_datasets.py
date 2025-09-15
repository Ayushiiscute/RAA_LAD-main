# combine_datasets.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, zipfile, tarfile, shutil, hashlib, gzip
from pathlib import Path
from typing import Iterable, Tuple

LOG_SUFFIXES = {".log", ".txt", ".out", ".csv"}
GZ_SUFFIX = ".gz"

def unzip_if_needed(src: Path, stage_dir: Path) -> Path:
    """Return a directory path containing files from src (zip or dir)."""
    if src.is_dir():
        return src

    stage_dir.mkdir(parents=True, exist_ok=True)
    root = stage_dir / src.stem
    root.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(root)
    elif tarfile.is_tarfile(src):
        with tarfile.open(src, "r:*") as tf:
            tf.extractall(root)
    else:
        raise ValueError(f"Unsupported archive: {src}")

    # If the extraction produced a single top-level folder, use it
    entries = [p for p in root.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root

def is_log_like(p: Path) -> bool:
    if p.suffix.lower() in LOG_SUFFIXES:
        return True
    # allow .log.gz / .txt.gz etc.
    if p.suffix.lower() == GZ_SUFFIX and p.with_suffix("").suffix.lower() in LOG_SUFFIXES:
        return True
    return False

def iter_log_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and is_log_like(p):
            yield p

def md5sum(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def copy_or_inflate(src_file: Path, src_root: Path, out_root: Path, origin: str,
                    seen_hashes: set) -> Tuple[bool, Path]:
    """Copy (or gunzip) a file into out_root/origin/<relative path>.
       Returns (copied, dest_path). Skips by file hash if already seen."""
    rel = src_file.relative_to(src_root)
    dest = out_root / origin / rel

    # Ensure parent dir exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Compute hash to avoid duplicate copies across datasets
    h = md5sum(src_file)
    if h in seen_hashes:
        return False, dest
    seen_hashes.add(h)

    # Decompress .gz to the same name without .gz
    if src_file.suffix.lower() == GZ_SUFFIX:
        inner = dest.with_suffix("")  # drop .gz
        inner.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(src_file, "rb") as fin, open(inner, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return True, inner

    # Regular copy
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dest)
    return True, dest

def main():
    ap = argparse.ArgumentParser(description="Combine two log datasets into one folder")
    ap.add_argument("--dataset-a", required=True, help="Path to first dataset (zip or dir)")
    ap.add_argument("--dataset-b", required=True, help="Path to second dataset (zip or dir)")
    ap.add_argument("--outdir", required=True, help="Where to write combined raw files")
    ap.add_argument("--name-a", default="dataset_a", help="Subfolder name for dataset A")
    ap.add_argument("--name-b", default="dataset_b", help="Subfolder name for dataset B")
    ap.add_argument("--stage", default=".stage_extract", help="Temp extraction area")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(args.stage)
    stage_dir.mkdir(parents=True, exist_ok=True)

    ds_a = Path(args.dataset_a)
    ds_b = Path(args.dataset_b)

    print("Extracting / resolving inputs...")
    a_root = unzip_if_needed(ds_a, stage_dir)
    b_root = unzip_if_needed(ds_b, stage_dir)
    print(f" - A root: {a_root}")
    print(f" - B root: {b_root}")

    seen_hashes = set()
    counts = {"copied": 0, "skipped": 0}

    def process(root: Path, origin: str):
        print(f"\nScanning {origin} ...")
        for f in iter_log_files(root):
            copied, dest = copy_or_inflate(f, root, out_root, origin, seen_hashes)
            if copied:
                counts["copied"] += 1
            else:
                counts["skipped"] += 1
        print(f"Done {origin}.")

    process(a_root, args.name_a)
    process(b_root, args.name_b)

    print("\n=== SUMMARY ===")
    print(f"Output folder: {out_root.resolve()}")
    print(f"Files copied : {counts['copied']}")
    print(f"Duplicates skipped: {counts['skipped']}")
    print("Now feed this folder to your preprocessing script, e.g.:")
    print(f"  python preprocess.py --input {out_root} --outdir artifacts_combined --save-csv")

if __name__ == "__main__":
    main()
