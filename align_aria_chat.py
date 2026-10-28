# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Studio 35: Aria Chat Alignment Fine-Tuner v2.6 Apex
================================================================================
【外科治療・過学習（ぎゅー一色モード崩壊）完全根絶 ✕ バランス浸透仕様】
1. Completion-Only Masking:
   - <start_of_turn>user ... <start_of_turn>model\n のプロンプト部分を -100 で厳格マスク！
   - 質問内容の条件付け（Conditioning）を強制し、無条件「ぎゅー」アトラクターを物理消滅！
2. Gradient Accumulation (有効バッチサイズ 4):
   - 4サンプルの勾配を集約して更新することで、単一サンプルの過剰局所解を防ぎ全体へ均等浸透！
3. Fast Gear (Loop=1) ＆ Deep Gear (Loop=2) 協調学習:
   - Fastモード（Loop=1）を主軸（80%）にしつつ、Deepモード（Loop=2）も20%注入！
4. 自動事後検品オラクル:
   - 学習終了直後に、実機主要プロンプト5件を自動推論して検品レポートを出力！
================================================================================
"""

import os
import sys
import time
import json
import random
import shutil
import argparse
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.aria_engraft_deltaloop import AriaEngraftDeltaLoopModel
from core.barron_softmax_ops import barron_cross_entropy_exact
from core.deltaloop_sampler import AriaDeltaLoopSampler

GEMMA4_ENGRAFT_DIR = os.path.join(current_dir, "models", "gemma4_engraft")
GEMMA4_BASE_WEIGHTS = os.path.join(GEMMA4_ENGRAFT_DIR, "gemma4_engraft_weights.pt")

DEFAULT_OMNI_PATH = os.path.join(current_dir, "models", "gemma4_rsi5_7_omni", "gemma4_rsi5_7_omni_weights.pt")
FALLBACK_ENGRAFT_PATH = os.path.join(current_dir, "models", "deltaloop_engraft_gemma4", "deltaloop_engraft_gemma4_weights.pt")
ALIGN_DATASET_PATH = os.path.join(current_dir, "dataset_cache", "aria_chat_alignment_dataset.jsonl")


def run_chat_alignment(weights_path: str = None, total_steps: int = 300, lr: float = 4.0e-5, grad_accum: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🌸 👑 ⚡ 【Studio 35: Aria Chat Alignment v2.6 Apex】 ⚡ 👑 🌸")
    print("=" * 80)
    print(f"🖥️ 実行デバイス: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")

    if weights_path is None:
        if os.path.exists(DEFAULT_OMNI_PATH):
            weights_path = DEFAULT_OMNI_PATH
            print(f"  💎 最新 Omni 全能重みをターゲットに設定: {os.path.basename(weights_path)}")
        elif os.path.exists(FALLBACK_ENGRAFT_PATH):
            weights_path = FALLBACK_ENGRAFT_PATH
            print(f"  💎 ENGRAFT 重みをターゲットに設定: {os.path.basename(weights_path)}")
        else:
            print("❌ 学習済み重みが見つかりません。")
            return

    backup_path = weights_path.replace(".pt", "_pre_chat_align.pt")
    if not os.path.exists(backup_path):
        print(f"  🛡️ 直前の重みを安全バックアップ: {os.path.basename(backup_path)}")
        shutil.copyfile(weights_path, backup_path)

    print(f"🎯 目標: ぎゅー一色モード崩壊の完全根絶 ＆ 自己紹介・エンジニア能力の均等浸透")
    print(f"⏱️ 更新ステップ: {total_steps} steps | 有効バッチ: {grad_accum} | 学習率: {lr:.2e}")
    print("-" * 80)

    # 1. トークナイザーとモデル準備
    print("\n[Step 1] トークナイザーとモデルのロード...")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA4_ENGRAFT_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = AriaEngraftDeltaLoopModel.load_from_gemma4_weights(
        weight_path=GEMMA4_BASE_WEIGHTS,
        d_model=512,
        num_layers=12,
        loop_steps=4
    ).to(device)

    print(f"  🔄 重みをロード中: {os.path.basename(weights_path)}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print("  ✅ 知識を完全継承！")

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

    # 2. アライメントデータセットのロード
    print("\n[Step 2] 純粋対話アライメントデータのロード...")
    corpus = []
    with open(ALIGN_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"  📂 総データ数: {len(corpus)} 件 (黄金バランス調整済み)")

    # 3. Completion-Only Masking ✕ 勾配累積 アライメントループ
    print(f"\n[Step 3] Completion-Only アライメント学習スタート！")
    t0 = time.perf_counter()
    running_loss = 0.0

    sample_idx = 0
    shuffled_corpus = list(corpus)
    random.seed(42)
    random.shuffle(shuffled_corpus)

    for step in range(1, total_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum):
            item = shuffled_corpus[sample_idx % len(shuffled_corpus)]
            sample_idx += 1
            if sample_idx % len(shuffled_corpus) == 0:
                random.shuffle(shuffled_corpus)

            inst = item["instruction"]
            resp = item["response"]

            prompt_text = f"<start_of_turn>user\n{inst}<end_of_turn>\n<start_of_turn>model\n"
            prompt_enc = tokenizer(prompt_text, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].size(1)

            full_text = f"{prompt_text}{resp}<end_of_turn>"
            full_enc = tokenizer(full_text, truncation=True, max_length=128, return_tensors="pt")
            input_ids = full_enc["input_ids"].to(device)

            if input_ids.size(1) <= prompt_len:
                continue

            labels = input_ids.clone()
            labels[:, :prompt_len] = -100

            # Fastモード（Loop=1）主軸（80%）、Deepモード（Loop=2）協調（20%）
            loop_s = 1 if random.random() < 0.8 else 2
            act_l = 4 if random.random() < 0.8 else 12

            logits, _, _ = model(input_ids, active_layers=act_l, loop_steps=loop_s)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = barron_cross_entropy_exact(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            accum_loss = loss / grad_accum
            accum_loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += (step_loss / grad_accum)

        if step % 25 == 0 or step == 10:
            avg_loss = running_loss / (25 if step > 10 else 10)
            running_loss = 0.0
            elapsed = time.perf_counter() - t0
            steps_per_sec = step / max(1e-5, elapsed)
            remaining_sec = (total_steps - step) / max(1e-5, steps_per_sec)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  Step [{step:3d}/{total_steps:3d}] | Loss: {avg_loss:.4f} | lr: {cur_lr:.1e} | ⚡ {steps_per_sec:.1f} steps/s | 残り: {remaining_sec:.1f}s")

    total_elapsed = time.perf_counter() - t0

    # 保存
    trainable_state = {k: v.cpu() for k, v in model.state_dict().items() if "embed_tokens" not in k}
    torch.save(trainable_state, weights_path)

    print("\n" + "=" * 80)
    print(f"🎉 🌸 【Completion-Only アライメント完了！】 総時間: {total_elapsed:.2f} 秒 🌸 🎉")
    print(f"💾 更新重み: {weights_path}")
    print("=" * 80)

    # 4. 事後自動検品オラクル
    print("\n[Step 4] 🌸 事後自動検品オラクル実行...")
    model.eval()
    sampler = AriaDeltaLoopSampler(model=model, device=device, temperature=0.7, stop_token_ids=[1, 3])

    test_prompts = [
        "こんにちわ",
        "アリアちゃんについて教えて",
        "アリアちゃんは何ができるの？",
        "自己紹介して！",
        "アリアちゃんのおっぱい触りたいです",
        "ぎゅーして！",
        "今日もお疲れ様でした！"
    ]

    print("-" * 80)
    for q in test_prompts:
        p = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        p_ids = tokenizer(p, return_tensors="pt")["input_ids"][0].tolist()
        res = sampler.generate_mtp(prompt_ids=p_ids, max_new_tokens=50, active_layers=4, loop_steps=1, physarum_gain=0.0)
        raw = tokenizer.decode(res["tokens"], skip_special_tokens=False)
        clean = raw.split("<end_of_turn>")[0].replace("<start_of_turn>model\n", "").strip()
        print(f"マスター > {q}")
        print(f"アリア   > {clean}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="Path to weights to fine-tune")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=4.0e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()
    run_chat_alignment(weights_path=args.weights, total_steps=args.steps, lr=args.lr, grad_accum=args.grad_accum)
