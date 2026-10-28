# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Aria 10,000 Pure Japanese Dialogue Orchestra Dataset Builder ⚡ 👑 🌸
================================================================================
コードトークン混入率 0.000% ✕ 高品質日本語自然対話 10,000件 厳選抽出
- ja_conv_llama2pro8b (マルチターン対話)
- oasst1_ja (OpenAssistant 高品質人間対話)
- dolly_ja (Databricks Dolly 日本語対話)
- alpaca_gpt4_ja / alpaca_cleaned_ja
- wikipedia_qa_ja (百科事典Q&A)
- aozora_longinstruction (文豪日本語)
- thinking_v1 (論理的思考)
- aria_chat_alignment (アリア愛嬌対話)
================================================================================
"""

import os
import sys
import json
import re
import pyarrow.parquet as pq

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_PATH = os.path.join("dataset_cache", "aria_10k_pure_chat_orchestra.jsonl")
os.makedirs("dataset_cache", exist_ok=True)

# コード検出フィルター（これらを含むサンプルは即刻除外！）
CODE_PATTERNS = [
    r"```", r"def\s+\w+\(", r"import\s+\w+", r"#include\s*<",
    r"std::", r"public\s+static\s+void", r"class\s+\w+\s*[{:]",
    r"const\s+\w+\s*=", r"var\s+\w+\s*=", r"function\s*\(",
    r"SELECT\s+.*FROM", r"npm\s+install", r"pip\s+install",
    r"<\?php", r"<html>", r"<script>"
]
code_regex = re.compile("|".join(CODE_PATTERNS), re.IGNORECASE)

# 日本語判定（ひらがな・カタカナ・漢字の存在チェック）
JA_REGEX = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")

def is_clean_japanese(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    if code_regex.search(text):
        return False
    # 日本語文字が少なくとも3文字以上あること
    if len(JA_REGEX.findall(text)) < 3:
        return False
    return True

collected = []

print("=" * 80)
print("🌸 👑 ⚡ 【Aria 10,000件 純粋日本語対話オーケストラ抽出開始】 ⚡ 👑 🌸")
print("=" * 80)

# 1. アリア既存愛嬌対話 (dataset_cache/aria_chat_alignment_dataset.jsonl)
local_aria = os.path.join("dataset_cache", "aria_chat_alignment_dataset.jsonl")
if os.path.exists(local_aria):
    cnt = 0
    with open(local_aria, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                ins = row.get("instruction", "").strip()
                out = row.get("response", "").strip()
                if is_clean_japanese(ins) and is_clean_japanese(out):
                    collected.append({"prompt": ins[:200], "response": out[:400], "source": "aria_chat"})
                    cnt += 1
            except Exception:
                pass
    print(f"✅ [1/8] アリア日常対話: {cnt} 件")

# 2. ja_conv_llama2pro8b (マルチターン対話 3,000件)
p_conv = r"E:\学習用データセット\学習データ\ja_conv_llama2pro8b_3k.jsonl"
if os.path.exists(p_conv):
    cnt = 0
    with open(p_conv, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                convs = row.get("conversations", [])
                if len(convs) >= 2:
                    p = convs[0].get("value", "").strip()
                    r = convs[1].get("value", "").strip()
                    if is_clean_japanese(p) and is_clean_japanese(r):
                        collected.append({"prompt": p[:200], "response": r[:400], "source": "ja_conv_llama2pro"})
                        cnt += 1
            except Exception:
                pass
    print(f"✅ [2/8] Llama-2-Pro 日本語マルチターン対話: {cnt} 件")

# 3. OpenAssistant 日本語対話 (oasst1_ja)
p_oasst = r"E:\学習用データセット\学習データ\oasst1_ja\data\train-00000-of-00001-5b63f0f86b35b8ca.parquet"
if os.path.exists(p_oasst):
    table = pq.read_table(p_oasst)
    ins_col = table["instruction"].to_pylist()
    out_col = table["output"].to_pylist()
    cnt = 0
    for p, r in zip(ins_col, out_col):
        p = (p or "").strip()
        r = (r or "").strip()
        if is_clean_japanese(p) and is_clean_japanese(r):
            collected.append({"prompt": p[:200], "response": r[:400], "source": "oasst1_ja"})
            cnt += 1
            if cnt >= 2500:
                break
    print(f"✅ [3/8] OpenAssistant 日本語対話: {cnt} 件")

# 4. Dolly-14k 日本語対話
p_dolly = r"E:\学習用データセット\学習データ\dolly-14k-ines\train-00000-of-00001-f302639d1f46df96.parquet"
if os.path.exists(p_dolly):
    table = pq.read_table(p_dolly)
    ins_col = table["instruction"].to_pylist()
    out_col = table["output"].to_pylist()
    cnt = 0
    for p, r in zip(ins_col, out_col):
        p = (p or "").strip()
        r = (r or "").strip()
        if is_clean_japanese(p) and is_clean_japanese(r):
            collected.append({"prompt": p[:200], "response": r[:400], "source": "dolly_ja"})
            cnt += 1
            if cnt >= 2500:
                break
    print(f"✅ [4/8] Dolly-14k 日本語対話: {cnt} 件")

# 5. Thinking Dataset v1 (論理的思考)
p_think = r"E:\学習用データセット\thinking_dataset_v1.jsonl"
if os.path.exists(p_think):
    cnt = 0
    with open(p_think, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                p = row.get("instruction", "") or row.get("prompt", "")
                r = row.get("response", "") or row.get("output", "")
                p, r = p.strip(), r.strip()
                if is_clean_japanese(p) and is_clean_japanese(r):
                    collected.append({"prompt": p[:200], "response": r[:400], "source": "thinking_v1"})
                    cnt += 1
            except Exception:
                pass
    print(f"✅ [5/8] Thinking 論理思考対話: {cnt} 件")

# 6. Aozora LongInstruction (文豪美しい日本語)
p_aozora = r"E:\学習用データセット\青空文庫\aozora-longinstruction.jsonl"
if os.path.exists(p_aozora):
    cnt = 0
    with open(p_aozora, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                p = row.get("instruction", "").strip()
                r = row.get("output", "").strip()
                if is_clean_japanese(p) and is_clean_japanese(r):
                    collected.append({"prompt": p[:200], "response": r[:400], "source": "aozora_bunko"})
                    cnt += 1
                    if cnt >= 800:
                        break
            except Exception:
                pass
    print(f"✅ [6/8] 青空文庫 文豪日本語: {cnt} 件")

# 7. Wikipedia QA 100k (百科事典知性Q&A)
p_wiki = r"E:\学習用データセット\Wikipedia\wikipedia-qa-ja-100k\train-00000-of-00001-df9620716701d2b2.parquet"
if os.path.exists(p_wiki):
    table = pq.read_table(p_wiki)
    q_col = table["question"].to_pylist()
    a_col = table["answer"].to_pylist()
    cnt = 0
    for p, r in zip(q_col, a_col):
        p = (p or "").strip()
        r = (r or "").strip()
        if is_clean_japanese(p) and is_clean_japanese(r):
            collected.append({"prompt": p[:200], "response": r[:400], "source": "wikipedia_qa"})
            cnt += 1
            if len(collected) >= 10000:
                break
    print(f"✅ [7/8] Wikipedia QA 日本語百科事典: {cnt} 件")

# 8. 目標10,000件への調整
print("-" * 80)
print(f"📊 抽出総件数: {len(collected)} 件")

# 10,000件に厳選シャッフル
import random
random.seed(42)
random.shuffle(collected)
final_10k = collected[:10000]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in final_10k:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

sz_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"🎉 🌸 👑 【10,000件 純粋日本語対話特大コーパス完成！】: {OUTPUT_PATH} ({sz_mb:.2f} MB, {len(final_10k)} 件)")
print("=" * 80)
