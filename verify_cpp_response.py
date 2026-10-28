# -*- coding: utf-8 -*-
import sys
import os
import torch
from transformers import AutoTokenizer

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
from core.cartridge_slot_manager import AriaCartridgeSlotManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEMMA4_ENGRAFT_DIR = os.path.join(current_dir, "models", "gemma4_engraft")
GEMMA4_BASE_WEIGHTS = os.path.join(GEMMA4_ENGRAFT_DIR, "gemma4_engraft_weights.pt")
TARGET_WEIGHTS = os.path.join(current_dir, "models", "gemma4_rsi5_7_omni", "gemma4_rsi5_7_omni_weights.pt")

tokenizer = AutoTokenizer.from_pretrained(GEMMA4_ENGRAFT_DIR)
model = AriaEngraftDeltaLoopModel.load_from_gemma4_weights(GEMMA4_BASE_WEIGHTS, d_model=512, num_layers=12, loop_steps=4).to(device)
state = torch.load(TARGET_WEIGHTS, map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)

sampler = AriaDeltaLoopSampler(
    model=model,
    device=device,
    temperature=0.7,
    top_k=40,
    repetition_penalty=1.15,
    eos_token_id=tokenizer.eos_token_id or 1,
    stop_token_ids=[1, 3]
)
cartridge_mgr = AriaCartridgeSlotManager(model, cartridges_dir=os.path.join(current_dir, "models", "cartridges"), device=device)

# Test query
query = "C++でコードを作れますか？"
docked = cartridge_mgr.auto_dock_by_prompt(query)
print(f"Query: {query}")
print(f"Docked cartridges: {docked}")

prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
enc = tokenizer(prompt, return_tensors="pt")
prompt_ids = enc["input_ids"][0].tolist()

res = sampler.generate_mtp(
    prompt_ids=prompt_ids,
    max_new_tokens=96,
    min_new_tokens=1,
    active_layers=12,
    loop_steps=2,
    physarum_gain=0.03
)

output_tokens = res["tokens"]
raw_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

# Clean
cleaned = raw_text
for prefix in ["<end_of_turn>", "<start_of_turn>model\n", "<start_of_turn>model", "<start_of_turn>"]:
    while cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):].strip()
cleaned = cleaned.strip()
for stop_mark in ["<end_of_turn>", "<start_of_turn>", "<turn|>", "ユーザー:", "マスター:"]:
    if stop_mark in cleaned:
        cleaned = cleaned.split(stop_mark)[0].strip()

print("\n=== Generated Response ===")
print(cleaned)
print("==========================")
print(f"Tokens: {res['tokens_count']} | Pure TPS: {res['pure_tps']:.1f} tok/s")
