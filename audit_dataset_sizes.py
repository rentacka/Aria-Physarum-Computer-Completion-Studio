# -*- coding: utf-8 -*-
import os
import sys
import json
import pyarrow.parquet as pq

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

datasets = [
    ("CodeAlpaca (C++, Python等)", r"E:\学習用データセット\学習データ\codealpaca-master\data\code_alpaca_20k.json"),
    ("Amenokaku pandas (データ分析)", r"E:\学習用データセット\学習データ\amenokaku_code_instruct.json"),
    ("MATH (数学・証明)", r"E:\学習用データセット\学習データ\MATH_Instruction_Format\train-00000-of-00001-587265c43a846637.parquet"),
    ("Linux Commands (Linux操作)", r"E:\学習用データセット\学習データ\instruction_linux_command.json"),
    ("LawQA 日本の法律 (金商法等)", r"E:\学習用データセット\lawqa_jp-main日本の法律\lawqa_jp-main\data\selection_with_reference__randomized.json"),
    ("Apollo 医療・生化学 (健康・薬理)", r"E:\学習用データセット\学習データ\医療系データセット\apollocorpus_ja_20240313.jsonl"),
    ("青空文庫 (近代文学・情緒表現)", r"E:\学習用データセット\青空文庫\aozora-longinstruction.jsonl"),
    ("Wikipedia QA 15k (百科事典)", r"E:\学習用データセット\Wikipedia\wikipedia-qa-ja-15k\train-00000-of-00001-effb97521c87f1e9.parquet"),
    ("Buffett 投資モデル (バフェット)", r"E:\学習用データセット\in-foxhound-ja(金融モデル、バフェット\train-00000-of-00001.parquet"),
    ("Thinking Dataset (CoT深層推論)", r"E:\学習用データセット\thinking_dataset_v1.jsonl"),
    ("Onomatopoeia (オノマトペ)", r"E:\学習用データセット\onomatopoeia-ja\train-00000-of-00001-697103d187735e15.parquet"),
]

print("-" * 75)
print(f"{'データセット名':<35} | {'ファイルサイズ':<12} | {'総件数':<12}")
print("-" * 75)

for name, p in datasets:
    if os.path.exists(p):
        size_bytes = os.path.getsize(p)
        if size_bytes >= 1024 * 1024 * 1024:
            size_str = f"{size_bytes / (1024**3):.2f} GB"
        else:
            size_str = f"{size_bytes / (1024**2):.2f} MB"
        
        count_str = "調査中"
        if p.endswith(".parquet"):
            try:
                t = pq.read_table(p)
                count_str = f"{t.num_rows:,} 件"
            except Exception:
                pass
        elif p.endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, list):
                    count_str = f"{len(d):,} 件"
                elif isinstance(d, dict) and "samples" in d:
                    count_str = f"{len(d['samples']):,} 件"
            except Exception:
                pass
        elif p.endswith(".jsonl"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                count_str = f"{lines:,} 件"
            except Exception:
                pass

        print(f"{name:<35} | {size_str:<12} | {count_str:<12}")
    else:
        print(f"{name:<35} | {'未検出':<12} | {'-':<12}")

print("-" * 75)
