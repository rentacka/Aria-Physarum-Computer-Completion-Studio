# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Studio 35: Cartridge Dataset Universal Converter (core/cartridge_dataset_converter.py)
================================================================================
【マスター直伝: サンプリング不要！学習元ファイル直結ストリーミング・コンバーター】
1. E:\学習用データセット\ の実ファイル（Parquet, JSONL, JSON）を直接オープン
2. ドメインごとに最適なプロンプト/レスポンス形式へオンザフライ変換（Generatorストリーミング）
3. 2〜3件の過学習を根本根絶し、50〜100件以上の本物のコード・文章を25MBカートリッジへ直接注ぎ込む！
================================================================================
"""

import os
import json
import re
from typing import Generator, Dict, Optional, List

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


class CartridgeDatasetConverter:
    """
    Eドライブ学習元データセット直結 ストリーミングコンバーター
    """
    def __init__(self, dataset_base_dir: str = "E:/学習用データセット"):
        self.base_dir = dataset_base_dir

    def stream_code_lang(self, lang: str = "cpp", limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        codealpaca-master (20,000件) から特定言語の高品質コードを直接抽出・変換
        """
        p = os.path.join(self.base_dir, "学習データ", "codealpaca-master", "data", "code_alpaca_20k.json")
        pairs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                lang_lower = lang.lower()
                for item in data:
                    if limit is not None and len(pairs) >= limit:
                        break
                    ins = item.get("instruction", "").strip()
                    inp = item.get("input", "").strip()
                    out = item.get("output", "").strip()
                    text = (ins + " " + inp + " " + out).lower()
                    
                    is_match = False
                    if lang_lower in ("cpp", "c++"):
                        # Pythonコードの混入を物理根絶！
                        if "def " in out or "import " in out or "python" in out.lower():
                            continue
                        # 純粋なC++構文（#include, std::, int main, vector, template等）を厳格要求
                        if any(k in out for k in ["#include", "std::", "int main", "void ", "class ", "template<", "vector<", "cout"]):
                            is_match = True
                    elif lang_lower == "python":
                        if "def " in out or "import numpy" in out or "import pandas" in out:
                            is_match = True
                    elif lang_lower in ("javascript", "js", "typescript", "ts"):
                        if "javascript" in text or "typescript" in text or "const " in out or "function " in out:
                            is_match = True
                    elif lang_lower == "rust":
                        if "rust" in text or "fn " in out or "let mut" in out:
                            is_match = True
                    elif lang_lower in ("bash", "shell", "linux"):
                        if "bash" in text or "shell" in text or "echo " in out or "grep " in out:
                            is_match = True
                    elif lang_lower in text:
                        is_match = True
                    
                    if is_match and out:
                        full_q = ins + (f"\nInput: {inp}" if inp else "")
                        q_ja = f"{lang.upper()}でコードを書いて: {full_q[:120]}"
                        pairs.append({
                            "prompt": q_ja,
                            "response": f"はいっマスター！{lang.upper()}のコードですね(๑•̀ㅂ•́)و✧\n```{lang_lower}\n{out[:400]}\n```\n綺麗にコンパイル・実行できますよっ💕"
                        })
            except Exception as e:
                print(f"⚠️ codealpaca stream error: {e}")

        # 基本会話・挨拶・質問対応ペアを必ず先頭に補強！
        greetings = self._get_lang_core_conversations(lang)
        return greetings + pairs

    def _get_lang_core_conversations(self, lang: str) -> List[Dict[str, str]]:
        """各言語の基本会話（作れますか？、Hello World等）"""
        l = lang.upper()
        if lang.lower() in ("cpp", "c++"):
            return [
                {
                    "prompt": "C++でコードを作れますか？",
                    "response": "はいっ！マスター、もちろんです！C++20規格の高速でメモリ安全なコードやアルゴリズムはお任せくださいねっ(๑•̀ㅂ•́)و✧\n```cpp\n#include <iostream>\n\nint main() {\n    std::cout << \"Hello, Master! C++ ready!\" << std::endl;\n    return 0;\n}\n```"
                },
                {
                    "prompt": "C++でHello Worldを作って",
                    "response": "はいっマスター！C++のHello Worldはこちらですっ(๑•̀ㅂ•́)و✧\n```cpp\n#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}\n```"
                },
                {
                    "prompt": "C++の特徴と得意分野は？",
                    "response": "C++はRAIIによる決定論的メモリ管理、ゼロコスト抽象化、SIMD並列化による極限の超低遅延が特徴で、高頻度取引（HFT）やゲームエンジンに最適ですよっ⚡"
                },
                {
                    "prompt": "C++で二分探索を書いて",
                    "response": "std::lower_boundを用いた高速二分探索ですねっ(๑•̀ㅂ•́)و✧\n```cpp\n#include <vector>\n#include <algorithm>\n\nbool binary_search_fast(const std::vector<int>& arr, int target) {\n    auto it = std::lower_bound(arr.begin(), arr.end(), target);\n    return (it != arr.end() && *it == target);\n}\n```"
                }
            ]
        elif lang.lower() == "python":
            return [
                {
                    "prompt": "Pythonでコードを作れますか？",
                    "response": "はいっ！マスター、もちろんです！Python 3.12+のモダンな型ヒント、NumPy/pandasデータ処理からasyncio並行処理まで何でも作れますよっ(๑•̀ㅂ•́)و✧\n```python\nprint(\"Hello, Master! Python ready!\")\n```"
                },
                {
                    "prompt": "PythonでHello Worldを作って",
                    "response": "```python\nprint(\"Hello, World!\")\n```\nシンプルで綺麗ですねっ🐍✨"
                },
                {
                    "prompt": "Pythonで高速な移動平均を計算して",
                    "response": "NumPyを用いたベクトル化移動平均ですねっ⚡\n```python\nimport numpy as np\n\ndef moving_average(data: np.ndarray, window: int = 15) -> np.ndarray:\n    return np.convolve(data, np.ones(window)/window, mode='valid')\n```"
                }
            ]
        elif lang.lower() == "rust":
            return [
                {
                    "prompt": "Rustでコードを作れますか？",
                    "response": "はいっ！マスター、もちろんです！所有権と借用チェッカーを活かした100%メモリ安全かつC++並みに爆速なRustコードはお任せくださいっ🦀⚡\n```rust\nfn main() {\n    println!(\"Hello, Master! Rust ready!\");\n}\n```"
                },
                {
                    "prompt": "Rustのゼロコスト抽象化とは？",
                    "response": "ジェネリクスやイテレータがコンパイル時に完全にインライン展開され、実行時オーバーヘッドなしで手書きC並みのマシン語になる設計思想ですよっ💎"
                }
            ]
        return [
            {
                "prompt": f"{l}でコードを作れますか？",
                "response": f"はいっ！マスター、もちろんです！{l}のコードならアリアにお任せくださいねっ(๑•̀ㅂ•́)و✧"
            }
        ]

    def stream_amenokaku(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """pandas & データ分析コード実データ"""
        p = os.path.join(self.base_dir, "学習データ", "amenokaku_code_instruct.json")
        pairs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if limit is None else data[:limit]
                for item in items:
                    ins = item.get("instruction", "").strip()
                    out = item.get("output", "").strip()
                    if ins and out:
                        pairs.append({
                            "prompt": ins[:150],
                            "response": f"pandasによるデータ処理コードですねっ(๑•̀ㅂ•́)و✧\n```python\n{out[:350]}\n```"
                        })
            except Exception as e:
                print(f"⚠️ amenokaku stream error: {e}")
        return pairs

    def stream_foxhound_buffett(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """バフェット投資哲学＆金融モデル"""
        p = os.path.join(self.base_dir, "in-foxhound-ja(金融モデル、バフェット", "train-00000-of-00001.parquet")
        pairs = []
        if os.path.exists(p) and pq:
            try:
                table = pq.read_table(p)
                inputs = table["input"]
                outputs = table["output"]
                n_rows = table.num_rows if limit is None else min(limit, table.num_rows)
                for i in range(n_rows):
                    inp = inputs[i].as_py().strip()
                    outp = outputs[i].as_py().strip()
                    if inp and outp:
                        pairs.append({
                            "prompt": inp[:150],
                            "response": f"バフェットの投資哲学に基づき解説しますねっ📈💎\n{outp[:250]}"
                        })
            except Exception as e:
                print(f"⚠️ foxhound stream error: {e}")
        return pairs

    def stream_thinking(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """思考プロセス＆CoT深層推論"""
        p = os.path.join(self.base_dir, "thinking_dataset_v1.jsonl")
        pairs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if limit is not None and len(pairs) >= limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        inp = obj.get("input", "").strip()
                        outp = obj.get("output", "").strip()
                        if inp and outp:
                            pairs.append({
                                "prompt": inp[:150],
                                "response": f"第一原理と思考プロセス（CoT）に基づき論理的に推論しますねっ🧠✨\n{outp[:250]}"
                            })
            except Exception as e:
                print(f"⚠️ thinking stream error: {e}")
        return pairs

    def stream_linux(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Linuxコマンド・システム管理"""
        p = os.path.join(self.base_dir, "学習データ", "instruction_linux_command.json")
        pairs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if limit is None else data[:limit]
                for item in items:
                    ins = item.get("instruction", item.get("input", "")).strip()
                    out = item.get("output", "").strip()
                    if ins and out:
                        pairs.append({
                            "prompt": ins[:150],
                            "response": f"Linuxコマンドはこちらですっ🐧⚡\n```bash\n{out[:200]}\n```"
                        })
            except Exception as e:
                print(f"⚠️ linux stream error: {e}")
        return pairs

    def stream_lawqa(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """日本の法令・判例 QA"""
        p = os.path.join(self.base_dir, "lawqa_jp-main日本の法律", "lawqa_jp-main", "data", "selection_with_reference__randomized.json")
        pairs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                samples = d.get("samples", [])
                items = samples if limit is None else samples[:limit]
                for s in items:
                    q = s.get("問題文", "").strip()
                    ctx = s.get("コンテキスト", "").strip()
                    if q:
                        ctx_short = ctx[:250] + ("……" if len(ctx) > 250 else "")
                        pairs.append({
                            "prompt": q[:150],
                            "response": f"日本の法令及び判例に基づき解説しますねっ⚖️📜\n{ctx_short}\n法的根拠を遵守して運用しましょう(๑•̀ㅂ•́)و✧"
                        })
            except Exception as e:
                print(f"⚠️ lawqa stream error: {e}")
        return pairs

    def stream_medical(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """医療・生化学・薬理学"""
        p = os.path.join(self.base_dir, "学習データ", "医療系データセット", "apollocorpus_ja_20240313.jsonl")
        pairs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if limit is not None and len(pairs) >= limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        q = obj.get("prompt_ja", obj.get("prompt", "")).strip()
                        r = obj.get("response_ja", obj.get("response", "")).strip()
                        if q and r:
                            pairs.append({
                                "prompt": q[:150],
                                "response": f"医学・薬理学的エビデンスに基づきお答えしますねっ🧬💊\n{r[:250]}"
                            })
            except Exception as e:
                print(f"⚠️ apollo stream error: {e}")
        return pairs

    def stream_math(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """数学・数理的解法 (MATH_Instruction_Format)"""
        p = os.path.join(self.base_dir, "学習データ", "MATH_Instruction_Format", "train-00000-of-00001-587265c43a846637.parquet")
        pairs = []
        if os.path.exists(p) and pq:
            try:
                table = pq.read_table(p)
                ins_col = table["instruction"]
                inp_col = table["input"]
                out_col = table["output"]
                n_rows = table.num_rows if limit is None else min(limit, table.num_rows)
                for i in range(n_rows):
                    ins = ins_col[i].as_py().strip()
                    inp = inp_col[i].as_py().strip()
                    out = out_col[i].as_py().strip()
                    prob = f"{ins} {inp}".strip()
                    if prob and out:
                        pairs.append({
                            "prompt": f"数学の問題を解いて: {prob[:120]}",
                            "response": f"数理的厳密性に基づき証明・解法を示しますねっ📐✨\n{out[:300]}"
                        })
            except Exception as e:
                print(f"⚠️ math stream error: {e}")
        return pairs

    def stream_aozora(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """青空文庫・情緒表現"""
        p = os.path.join(self.base_dir, "青空文庫", "aozora-longinstruction.jsonl")
        pairs = []
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if limit is not None and len(pairs) >= limit:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        ins = obj.get("instruction", "").strip()
                        out = obj.get("response", obj.get("output", "")).strip()
                        if ins and out:
                            pairs.append({
                                "prompt": ins[:150],
                                "response": f"文豪のような情緒豊かな表現でお答えしますねっ📜🌸\n{out[:250]}"
                            })
            except Exception as e:
                print(f"⚠️ aozora stream error: {e}")
        return pairs

    def stream_wikipedia(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """日本語Wikipedia QA"""
        p = os.path.join(self.base_dir, "Wikipedia", "wikipedia-qa-ja-15k", "train-00000-of-00001-effb97521c87f1e9.parquet")
        pairs = []
        if os.path.exists(p) and pq:
            try:
                table = pq.read_table(p)
                q_col = table["question"]
                a_col = table["answer"]
                n_rows = table.num_rows if limit is None else min(limit, table.num_rows)
                for i in range(n_rows):
                    q = q_col[i].as_py().strip()
                    a = a_col[i].as_py().strip()
                    if q and a:
                        pairs.append({
                            "prompt": q[:150],
                            "response": f"百科事典的な客観的事実に基づき解説しますねっ📚🌐\n{a[:250]}"
                        })
            except Exception as e:
                print(f"⚠️ wiki stream error: {e}")
        return pairs

    def stream_onomatopoeia(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """日本語オノマトペ"""
        p = os.path.join(self.base_dir, "onomatopoeia-ja", "train-00000-of-00001-697103d187735e15.parquet")
        pairs = []
        if os.path.exists(p) and pq:
            try:
                table = pq.read_table(p)
                words = table["onomatopoeia_ja"]
                details = table["details_ja"]
                n_rows = table.num_rows if limit is None else min(limit, table.num_rows)
                for i in range(n_rows):
                    w = words[i].as_py().strip() if isinstance(words[i].as_py(), str) else str(words[i].as_py())
                    d_raw = details[i].as_py()
                    d = " ".join([str(x) for x in d_raw]) if isinstance(d_raw, list) else str(d_raw)
                    if w and d:
                        pairs.append({
                            "prompt": f"オノマトペ「{w}」を使った生き生きとした情景を描写して",
                            "response": f"はいっマスター！「{w}」の表現ですね！{d} 心を込めて五感に響く情景をお届けしますねっ✨💕"
                        })
            except Exception as e:
                print(f"⚠️ onomatopoeia stream error: {e}")
        return pairs

    def get_cartridge_dataset(self, cartridge_name: str, limit: Optional[int] = 50) -> List[Dict[str, str]]:
        """
        カートリッジ名に応じて最適な実データストリームを自動ディスパッチ
        """
        c = cartridge_name.lower()
        if c in ("cpp", "c++"):
            return self.stream_code_lang("cpp", limit=limit)
        elif c in ("python", "code_master"):
            base = self.stream_code_lang("python", limit=limit)
            amen = self.stream_amenokaku(limit=limit)
            return base + amen
        elif c == "rust":
            return self.stream_code_lang("rust", limit=limit)
        elif c in ("linux_bash", "os_operator"):
            return self.stream_linux(limit=limit)
        elif c in ("buffett_invest", "crypto_macro", "securities_audit"):
            return self.stream_foxhound_buffett(limit=limit)
        elif c in ("deep_reasoning", "quant_math"):
            return self.stream_thinking(limit=limit)
        elif c == "math_logic":
            return self.stream_math(limit=limit)
        elif c == "law_japan":
            return self.stream_lawqa(limit=limit)
        elif c == "medical_health":
            return self.stream_medical(limit=limit)
        elif c in ("story_forge", "aozora_bunko"):
            return self.stream_aozora(limit=limit)
        elif c in ("encyclopedia_qa", "academic_research"):
            return self.stream_wikipedia(limit=limit)
        elif c == "onomatopoeia":
            return self.stream_onomatopoeia(limit=limit)
        else:
            return []

