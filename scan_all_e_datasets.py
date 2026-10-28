import os
import sys

base_dir = r"E:\学習用データセット"
exts = {".json", ".jsonl", ".zip", ".zst", ".gz", ".parquet"}

print(f"=== Scanning {base_dir} for dataset files ===")
found = []
for root, dirs, files in os.walk(base_dir):
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() in exts:
            p = os.path.join(root, f)
            sz = os.path.getsize(p)
            found.append((p, sz))

found.sort(key=lambda x: x[1], reverse=True)
for p, sz in found:
    mb = sz / (1024 * 1024)
    print(f"{mb:8.2f} MB | {p}")
