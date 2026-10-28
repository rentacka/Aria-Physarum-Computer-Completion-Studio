# -*- coding: utf-8 -*-
import sys
import os
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.aria_engraft_deltaloop import AriaEngraftDeltaLoopModel
from core.deltaloop_sampler import AriaDeltaLoopSampler
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("models/gemma4_engraft")
model = AriaEngraftDeltaLoopModel.load_from_gemma4_weights(
    "models/gemma4_engraft/gemma4_engraft_weights.pt",
    d_model=512,
    num_layers=12,
    loop_steps=4
).to(device)

state = torch.load("models/gemma4_rsi5_7_omni/gemma4_rsi5_7_omni_weights.pt", map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()

sampler = AriaDeltaLoopSampler(model=model, device=device, temperature=0.7, stop_token_ids=[1, 3])

print("=" * 80)
print("🌸 【Aria 実機 TPS 測定ベンチマーク】")
print("=" * 80)

bench_configs = [
    ("⚡ 神速 FAST (日常会話: 4層 x 1Loop)", 4, 1),
    ("🧠 熟考 DEEP (クオンツ/コード: 12層 x 2Loop)", 12, 2),
    ("🚀 フルレイヤー (12層 x 1Loop)", 12, 1)
]

p = "<start_of_turn>user\nこんにちは！自己紹介して！<end_of_turn>\n<start_of_turn>model\n"
p_ids = tokenizer(p, return_tensors="pt")["input_ids"][0].tolist()

for name, layers, loop in bench_configs:
    # ウォームアップ
    _ = sampler.generate_mtp(prompt_ids=p_ids, max_new_tokens=10, active_layers=layers, loop_steps=loop, physarum_gain=0.0)
    # 本番計測
    res = sampler.generate_mtp(prompt_ids=p_ids, max_new_tokens=60, active_layers=layers, loop_steps=loop, physarum_gain=0.0)
    tps = res["pure_tps"]
    toks = res["tokens_count"]
    sec = res["total_elapsed_sec"]
    print(f"・{name}")
    print(f"    実効TPS: {tps:.1f} tok/s | 生成: {toks} tokens | 時間: {sec:.3f} 秒")

print("=" * 80)
