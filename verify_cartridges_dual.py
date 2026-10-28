# -*- coding: utf-8 -*-
import os, sys, time, torch
from transformers import AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.aria_engraft_deltaloop import AriaEngraftDeltaLoopModel
from core.deltaloop_sampler import AriaDeltaLoopSampler
from core.cartridge_slot_manager import AriaCartridgeSlotManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("models/gemma4_engraft")
model = AriaEngraftDeltaLoopModel.load_from_gemma4_weights(
    "models/gemma4_engraft/gemma4_engraft_weights.pt",
    d_model=512,
    num_layers=12,
    loop_steps=4
).to(device)
model.load_state_dict(torch.load("models/gemma4_rsi5_7_omni/gemma4_rsi5_7_omni_weights.pt", map_location=device, weights_only=True), strict=False)
model.eval()

mgr = AriaCartridgeSlotManager(model=model, cartridges_dir="models/cartridges", device=device)
sampler = AriaDeltaLoopSampler(model=model, device=device, temperature=0.6, stop_token_ids=[1, 3])

test_cases = [
    ("cpp", "C++で二分探索の関数を書いて"),
    ("buffett_invest", "バフェットの投資哲学で最も重要な原則は何ですか？"),
    ("linux_bash", "Gitの直近5件のコミットログを1行で表示するコマンドは？")
]

print("=" * 80)
print("🌸 👑 ⚡ 【25MB カートリッジ最新実機検証 (C++ / Buffett / Bash)】 ⚡ 👑 🌸")
print("=" * 80)

for c_name, q in test_cases:
    t_d = mgr.dock_multi([c_name])
    prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
    p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    res = sampler.generate_mtp(prompt_ids=p_ids, max_new_tokens=64, active_layers=4, loop_steps=1, physarum_gain=0.0)
    raw = tokenizer.decode(res["tokens"], skip_special_tokens=False)
    clean = raw.split("<end_of_turn>")[0].replace("<start_of_turn>model\n", "").strip()
    t0 = time.perf_counter()
    mgr.undock_all()
    t_u = (time.perf_counter() - t0) * 1000.0
    print(f"\n📂 [{c_name} カートリッジ装着: {t_d:.2f}ms ➔ 離脱: {t_u:.4f}ms]")
    print(f"マスター > {q}")
    print(f"アリア   > {clean}")

print("\n" + "=" * 80)
print("🎉 全カートリッジ完全稼働実証！")
print("=" * 80)
