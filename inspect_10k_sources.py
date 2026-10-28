import os
import sys
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = r"E:\学習用データセット"

try:
    import pyarrow.parquet as pq
    has_pq = True
except Exception:
    has_pq = False

def test_file(desc, path):
    if not os.path.exists(path):
        print(f"❌ {desc}: Not Found ({path})")
        return None
    sz = os.path.getsize(path) / (1024 * 1024)
    print(f"✅ {desc}: Found ({sz:.2f} MB)")
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
            print(f"   Keys: {list(first.keys())}")
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"   Count: {len(data)}, Keys: {list(data[0].keys())}")
            elif isinstance(data, dict):
                print(f"   Dict keys: {list(data.keys())[:5]}")
    elif path.endswith(".parquet") and has_pq:
        table = pq.read_table(path)
        print(f"   Count: {len(table)}, Columns: {table.column_names}")
    return path

print("=== Inspecting 10k candidate sources ===")
test_file("Llama2-Pro Conv", os.path.join(base, "学習データ", "ja_conv_llama2pro8b_3k.jsonl"))
test_file("Dolly-14k", os.path.join(base, "学習データ", "dolly-14k-ines", "train-00000-of-00001-f302639d1f46df96.parquet"))
test_file("OpenAssistant Ja", os.path.join(base, "学習データ", "oasst1_ja", "data", "train-00000-of-00001-5b63f0f86b35b8ca.parquet"))
test_file("Wikipedia QA 100k", os.path.join(base, "Wikipedia", "wikipedia-qa-ja-100k", "train-00000-of-00001-df9620716701d2b2.parquet"))
test_file("Amenokaku Synthetic", os.path.join(base, "amenokaku_qwen2.5_32b_synthetic.json"))
test_file("Thinking v1", os.path.join(base, "instruction_reasoning_thinking_dataset_v1.jsonl"))
test_file("Aozora LongInstruction", os.path.join(base, "aozora-longinstruction.jsonl"))
test_file("Onomatopoeia", os.path.join(base, "instruction_sensory_onomatopoeia_imagery.json"))
