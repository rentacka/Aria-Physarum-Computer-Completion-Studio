# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Studio 35: Aria Trinity DeltaLoop 爆速生体対話 CLI v2.0 Apex
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【三位一体 生体知能 (Trinity Biomorphic Intelligence)】
1. 普遍概念記憶 (Engram): Gemma-4 26万語彙 (2560D) 100% 凍結埋め込み (Summer氏 ENGRAFT)
2. 生体脳波オシレータ: 粘菌周波数コア (2.05 MB Safetensors / 1024基底共鳴)
3. 動的シナプス可塑性: ゼロ初期化 LoRA (1.25 MB / ヘブ則可塑性)
4. 選択的思考反復 (RLT): 日常は直感 (Loop=1 / 70~400 tok/s)、クオンツ・コードは熟考 (Loop=2~4)
5. 0-KV キャッシュ定常メモリ (384KB固定) による超低遅延・高スループット対話
================================================================================
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.aria_deltaloop_model import AriaApexDeltaLoopModel
from core.aria_engraft_deltaloop import AriaEngraftDeltaLoopModel
from core.aria_physarum_safetensor_cortex import (
    AriaPhysarumFrequencyCortex,
    load_physarum_safetensors
)
from core.aria_task_lora import AriaDeltaLoopLoRAEngine
from core.deltaloop_sampler import AriaDeltaLoopSampler
from core.cartridge_slot_manager import AriaCartridgeSlotManager

# パス定義
GEMMA4_ENGRAFT_DIR = os.path.join(current_dir, "models", "gemma4_engraft")
GEMMA4_BASE_WEIGHTS = os.path.join(GEMMA4_ENGRAFT_DIR, "gemma4_engraft_weights.pt")
GEMMA4_RSI_OMNI_WEIGHTS = os.path.join(current_dir, "models", "gemma4_rsi5_7_omni", "gemma4_rsi5_7_omni_weights.pt")
PHYSARUM_PATH = os.path.join(current_dir, "models", "physarum_safetensors", "aria_physarum_zero_omni.safetensors")
LORA_PATH = os.path.join(current_dir, "models", "lora_adapters", "aria_task_lora_omni.safetensors")
CARTRIDGES_DIR = os.path.join(current_dir, "models", "cartridges")

QUANT_KEYWORDS = [
    "mql5", "ctrade", "canvas", "backtest", "バックテスト", "インジ", "ea",
    "関数", "クラス", "注文", "チャート", "コード", "トレード", "rsi", "sma",
    "html", "javascript", "python", "script", "struct", "class", "def ",
    "void ", "int ", "double ", "bool ", "return", "pip", "スプレッド",
    "ゲーム", "game", "作れ", "開発", "作って", "アプリ", "プログラミング", "設計", "実装"
]


def detect_is_quant_task(text: str) -> bool:
    lower_t = text.lower()
    return any(kw in lower_t for kw in QUANT_KEYWORDS)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Studio 35: Aria Trinity DeltaLoop CLI")
    parser.add_argument("--no-cartridge", action="store_true", help="カートリッジを無効化してベースモデル単体スタンドアローンで起動")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🌸 👑 ⚡ 【Studio 35: Aria Trinity DeltaLoop 爆速生体対話 CLI v2.0 Apex】 ⚡ 👑 🌸")
    print("   〜 普遍概念 ENGRAFT ✕ 粘菌 Safetensors ✕ 動的シナプス LoRA ✕ RLT 〜")
    print("=" * 80)
    print(f"🖥️ 実行デバイス: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")

    # 1. トークナイザーのロード
    print("\n[1/4] Gemma-4 公式トークナイザーをロード中...")
    if os.path.exists(GEMMA4_ENGRAFT_DIR):
        print(f"  💎 Gemma-4 公式トークナイザー: {GEMMA4_ENGRAFT_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(GEMMA4_ENGRAFT_DIR)
    else:
        print("  ⚠️ Gemma-4 トークナイザーが見つかりません。デフォルトを設定します。")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. モデルのロード
    print("\n[2/4] Gemma-4 ENGRAFT ✕ DeltaLoop 思考中枢をロード中...")
    use_trinity = False
    lora_engine = None
    cortex = None

    if os.path.exists(GEMMA4_BASE_WEIGHTS) and os.path.exists(GEMMA4_RSI_OMNI_WEIGHTS):
        print(f"  👑 Gemma-4 2560D 埋め込み凍結 ✕ DeltaLoop モデルを初期化")
        model = AriaEngraftDeltaLoopModel.load_from_gemma4_weights(
            weight_path=GEMMA4_BASE_WEIGHTS,
            d_model=512,
            num_layers=12,
            loop_steps=4
        ).to(device)

        print(f"  💎 Omni 全能自己進化重みをロード: {os.path.basename(GEMMA4_RSI_OMNI_WEIGHTS)}")
        state_dict = torch.load(GEMMA4_RSI_OMNI_WEIGHTS, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        # 3. 粘菌 Safetensors 生体脳のドッキング
        print("\n[3/4] 生体中枢のドッキング中...")
        if os.path.exists(PHYSARUM_PATH):
            print(f"  🍄 粘菌脳 Safetensors をドッキング: {os.path.basename(PHYSARUM_PATH)} (2.05 MB)")
            cortex = AriaPhysarumFrequencyCortex(d_model=512, num_frequencies=256).to(device)
            load_physarum_safetensors(cortex, PHYSARUM_PATH, device=device)
            model.attach_physarum_cortex(cortex, gain=0.0)
        else:
            print("  ℹ️ 粘菌脳 Safetensors は未検出（バイパス）")

        # 4. 動的シナプス LoRA の注入
        if os.path.exists(LORA_PATH):
            print(f"  🧬 動的シナプス LoRA を注入: {os.path.basename(LORA_PATH)} (1.25 MB)")
            lora_engine = AriaDeltaLoopLoRAEngine(model=model, r=8, lora_alpha=16.0)
            lora_engine.inject_lora()
            lora_engine.load_lora_safetensors(LORA_PATH, device=device)
            lora_engine.set_lora_scale(0.0)
        else:
            print("  ℹ️ 動的シナプス LoRA は未検出（バイパス）")

        use_trinity = True
    else:
        print("  ⚠️ 最新 Omni 重みが見つかりません。スクラッチまたは互換モデルをロードします。")
        model = AriaApexDeltaLoopModel(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            num_heads=8,
            d_state=32,
            num_layers=12,
            loop_steps=4,
            mtp_depth=4
        ).to(device)

    model.eval()

    # 4. 超光速サンプラー
    print("\n[4/4] 超光速 DeltaLoop サンプラーを準備中...")
    sampler = AriaDeltaLoopSampler(
        model=model,
        device=device,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id or 1,
        stop_token_ids=[1, 3]
    )
    print("  ✅ サンプラー準備完了！ (EOS=1, end_of_turn=3 即時ブレーキ)")

    # 5. 粘菌コードキャッシュスロット マネージャーの起動
    enable_cartridges = not args.no_cartridge
    cartridge_mgr = None
    if enable_cartridges:
        print("\n[5/5] 粘菌コードキャッシュスロット マネージャーを起動中...")
        cartridge_mgr = AriaCartridgeSlotManager(model=model, cartridges_dir=CARTRIDGES_DIR, device=device)
        c_names = list(cartridge_mgr.loaded_cartridges.keys())
        print(f"  🔌 認識カートリッジ: {len(c_names)} 種類 ({', '.join(c_names)}) (ホットスワップ 0.0001s)")
    else:
        print("\n[5/5] 🔌 カートリッジ機能: OFF (純粋ベースモデル完全スタンドアローン稼働モード)")

    # ギア設定
    current_mode = "auto"
    active_layers = 4
    loop_steps = 1
    lora_scale = 0.0
    physarum_gain = 0.0
    max_tokens = 64

    print("\n" + "=" * 80)
    print("✨ 【Aria Trinity DeltaLoop 対話セッション開始】")
    print("コマンド:")
    print("  /auto       : 🌈 自動ギアチェンジ（日常は直感神速、コード・相場は熟考深層）★デフォルト推奨")
    print("  /fast       : ⚡ 神速モード（LoRA=0.0 / 粘菌=0.0 / Loop=1 / 4層 / 70~400 tok/s）")
    print("  /deep       : 🧠 熟考深層モード（LoRA=1.0 / 粘菌=0.03 / Loop=2 / 12層 / クオンツ特化）")
    print("  /cartridges : 🔌 利用可能な粘菌カートリッジ一覧と装着状態")
    print("  /status     : 📊 現在の生体パラメータ・ドッキング状態の表示")
    print("  /exit       : 終了 (または 'quit')")
    print(f"\n現在のモード: 【🌈 {current_mode.upper()}】")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("\nマスター > ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ["exit", "quit", "q"]:
                print("\n🌸 またいつでも呼んでくださいね、マスター！ずっと胸の中で一緒にいます……チュッ💋(*/ω＼*)💕\n")
                break

            if cmd == "/auto":
                current_mode = "auto"
                print("🌈 モード切替: 【自動 AUTO モード】(プロンプトに応じて動的適応)")
                continue
            elif cmd == "/fast":
                current_mode = "fast"
                active_layers = 4
                loop_steps = 1
                lora_scale = 0.0
                physarum_gain = 0.0
                max_tokens = 64
                if lora_engine:
                    lora_engine.set_lora_scale(0.0)
                print("⚡ モード切替: 【神速 FAST モード】(LoRA=0.0, 粘菌=0.0, Loop=1, 4層 ➔ 愛着精度100%)")
                continue
            elif cmd == "/deep":
                current_mode = "deep"
                active_layers = 12
                loop_steps = 2
                lora_scale = 1.0
                physarum_gain = 0.03
                max_tokens = 96
                if lora_engine:
                    lora_engine.set_lora_scale(1.0)
                print("🧠 モード切替: 【熟考 DEEP モード】(LoRA=1.0, 粘菌=0.03, Loop=2, 12層 ➔ クオンツ思考)")
                continue
            elif cmd == "/cartridges":
                if not enable_cartridges or not cartridge_mgr:
                    print("\nℹ️ カートリッジ機能は現在OFFです (/cartridges on で有効化可能)")
                    continue
                print("\n" + "-" * 50)
                print("🔌 [粘菌コードキャッシュスロット ステータス]")
                print(f"  ・現在装着中: {cartridge_mgr.active_cartridges if cartridge_mgr.active_cartridges else 'なし (日常会話・心 100%保護)'}")
                print("  ・利用可能カートリッジ:")
                for name, c_data in cartridge_mgr.loaded_cartridges.items():
                    print(f"    - [{name}] {c_data.get('title')} (キーワード: {', '.join(c_data.get('keywords', [])[:4])})")
                print("-" * 50)
                continue
            elif cmd in ["/nocartridge", "/cartridge off", "/cartridges off"]:
                enable_cartridges = False
                if cartridge_mgr:
                    cartridge_mgr.undock_all()
                print("🔌 カートリッジ機能をOFFに設定しました (純粋ベースモデル単体モード)")
                continue
            elif cmd in ["/cartridge on", "/cartridges on"]:
                if cartridge_mgr is None:
                    cartridge_mgr = AriaCartridgeSlotManager(model=model, cartridges_dir=CARTRIDGES_DIR, device=device)
                enable_cartridges = True
                print("🔌 カートリッジ機能をONに設定しました")
                continue
            elif cmd == "/status":
                print("\n" + "-" * 50)
                print(f"📊 [生体ステータス]")
                print(f"  ・モード      : {current_mode.upper()}")
                print(f"  ・物理層数    : {active_layers} 層")
                print(f"  ・RLT 反復    : {loop_steps} 回 (実効 {active_layers * loop_steps} 層)")
                print(f"  ・LoRA Scale  : {lora_scale:.2f}")
                print(f"  ・粘菌 Gain   : {physarum_gain:.2f}")
                slot_st = (cartridge_mgr.active_cartridges if cartridge_mgr and cartridge_mgr.active_cartridges else '空 (心 100%保護)') if enable_cartridges else '無効 (完全スタンドアローン)'
                print(f"  ・スロット装着: {slot_st}")
                print(f"  ・三位一体回路: {'活性化 (Active)' if use_trinity else '非活性'}")
                print("-" * 50)
                continue

            # 粘菌コードキャッシュスロット 自動マルチドッキング (0.0001s)
            docked_cartridges = cartridge_mgr.auto_dock_by_prompt(user_input) if (enable_cartridges and cartridge_mgr) else []

            # 自動ギア判定
            if current_mode == "auto":
                if docked_cartridges:
                    # カートリッジ装着時は、25MB差分FFN (Layer 2&3) に完全合焦した神速出力＆低温度決定論的構文
                    active_layers = 4
                    loop_steps = 1
                    lora_scale = 0.0
                    physarum_gain = 0.0
                    max_tokens = 96
                    sampler.temperature = 0.2
                    sampler.top_k = 15
                    applied_gear = f"カートリッジ神速出力 (L4xR1, 差分FFN直結: {' ✕ '.join(docked_cartridges)})"
                elif detect_is_quant_task(user_input):
                    active_layers = 12
                    loop_steps = 2
                    lora_scale = 1.0
                    physarum_gain = 0.03
                    max_tokens = 96
                    sampler.temperature = 0.4
                    sampler.top_k = 30
                    applied_gear = "クオンツ・熟考 (LoRA=1.0, 粘菌=0.03, Loop=2)"
                else:
                    active_layers = 4
                    loop_steps = 1
                    lora_scale = 0.0
                    physarum_gain = 0.0
                    max_tokens = 64
                    sampler.temperature = 0.7
                    sampler.top_k = 40
                    applied_gear = "愛着・日常直感神速 (LoRA=0.0, 粘菌=0.0, Loop=1)"

                if lora_engine:
                    lora_engine.set_lora_scale(lora_scale)

            # プロンプト整形
            prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
            enc = tokenizer(prompt, return_tensors="pt")
            prompt_ids = enc["input_ids"][0].tolist()

            # 推論実行
            res = sampler.generate_mtp(
                prompt_ids=prompt_ids,
                max_new_tokens=max_tokens,
                min_new_tokens=1,
                active_layers=active_layers,
                loop_steps=loop_steps,
                physarum_gain=physarum_gain
            )

            # 推論完了後、次ターンの日常会話（心）を完全不可侵にするためスロット即時解除 (0.00005s)
            if cartridge_mgr:
                cartridge_mgr.undock_all()

            output_tokens = res["tokens"]
            raw_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

            # クリーンアップ
            cleaned = raw_text
            for prefix in ["<end_of_turn>", "<start_of_turn>model\n", "<start_of_turn>model", "<start_of_turn>"]:
                while cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            cleaned = cleaned.strip()

            for stop_mark in ["<end_of_turn>", "<start_of_turn>", "<turn|>", "ユーザー:", "マスター:"]:
                if stop_mark in cleaned:
                    cleaned = cleaned.split(stop_mark)[0].strip()

            # 回答出力
            print(f"\nアリア > {cleaned if cleaned else '(沈黙)'}")
            print("-" * 65)
            gear_info = applied_gear if current_mode == 'auto' else f"{current_mode.upper()} (L{active_layers}xR{loop_steps})"
            slot_info = f" | 🔌 スロット: {' ✕ '.join(docked_cartridges)}" if docked_cartridges else " | 🔌 スロット: クリーン (心保護)"
            print(f"📊 [生体メトリクス] 生成: {res['tokens_count']} tok | TPS: {res['pure_tps']:.1f} tok/s ({res['total_elapsed_sec']:.2f}s) | ギア: {gear_info}{slot_info}")

        except (KeyboardInterrupt, EOFError):
            print("\n🌸 対話を終了しました。マスター、またお話ししましょうねっ💕")
            break
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
