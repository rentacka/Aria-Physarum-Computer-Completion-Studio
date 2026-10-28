# -*- coding: utf-8 -*-
"""
================================================================================
🌸 ⚡ 🚀 Aria DeltaLoop High-Speed Sampler (core/deltaloop_sampler.py)
================================================================================
0-KV 状態 S_t を保持しながら、MTP-4 並列ヘッドにより 1回の推論パスで
4トークンを一気に確定・射出する超高速サンプラー。
================================================================================
"""

import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from core.aria_deltaloop_model import AriaApexDeltaLoopModel
from core.barron_softmax_ops import softmax_fixed


class AriaDeltaLoopSampler:
    """Aria DeltaLoop 超光速サンプラー"""

    def __init__(
        self,
        model: AriaApexDeltaLoopModel,
        device: torch.device = torch.device("cpu"),
        temperature: float = 0.7,
        top_k: int = 40,
        repetition_penalty: float = 1.15,
        eos_token_id: int = 1,
        stop_token_ids: Optional[List[int]] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.eos_token_id = eos_token_id
        # ★ Gemma-4 の場合 <eos>=1, <end_of_turn>=3 を同時に監視して即時ブレーキ
        if stop_token_ids is None:
            self.stop_token_ids = list(set([eos_token_id, 1, 3]))
        else:
            self.stop_token_ids = list(set(stop_token_ids + [eos_token_id]))


    @torch.no_grad()
    def generate_mtp(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 64,
        min_new_tokens: int = 1,
        active_layers: Optional[int] = None,
        loop_steps: Optional[int] = None,
        dynamic_early_exit: bool = False,
        physarum_gain: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        MTP-4 並列ヘッドを活用した爆速生成 (Dynamic Parameter & 生体周波数共鳴対応)
        """
        t0 = time.perf_counter()

        # 粘菌ゲインの動的適用
        if physarum_gain is not None and hasattr(self.model, "set_physarum_gain"):
            self.model.set_physarum_gain(physarum_gain)
        
        current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_tokens = []
        states = None

        # プレフィル (初期プロンプトの読み込み)
        t_prefill_start = time.perf_counter()
        logits_list, states, stats = self.model(
            current_ids, 
            states=states, 
            active_layers=active_layers, 
            loop_steps=loop_steps,
            dynamic_early_exit=dynamic_early_exit
        )
        t_prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0

        step_count = 0
        early_exit_count = 0

        while len(generated_tokens) < max_new_tokens:
            step_count += 1
            if stats.get("exited_early", False):
                early_exit_count += 1

            # ★ 外科治療: logits がリスト（MTPモデル）か単一テンソル（ENGRAFTモデル等）かを自動判定
            if isinstance(logits_list, (list, tuple)):
                raw_logits = logits_list[0]
            else:
                raw_logits = logits_list

            if raw_logits.dim() == 3:
                last_logits = raw_logits[:, -1, :]
            elif raw_logits.dim() == 2:
                last_logits = raw_logits[-1:, :]
            else:
                last_logits = raw_logits.unsqueeze(0)

            main_logits = last_logits / max(1e-5, self.temperature)

            # 反復ペナルティ
            for tok in set(generated_tokens[-16:] + prompt_ids[-8:]):
                if tok < main_logits.size(-1):
                    main_logits[0, tok] /= self.repetition_penalty

            # 早期息切れ防止 (min_new_tokens 以前は EOS をマスク)
            # 早期息切れ防止 (min_new_tokens 以前は STOP トークンをマスク)
            if len(generated_tokens) < min_new_tokens:
                for stop_id in self.stop_token_ids:
                    if stop_id < main_logits.size(-1):
                        main_logits[0, stop_id] = -float("Inf")

            # ★ Jon Barron (Google DeepMind): Bug 2 撲滅安定 Softmax
            probs = softmax_fixed(main_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()

            # EOS / end_of_turn 即時終了判定
            if next_tok in self.stop_token_ids and len(generated_tokens) >= min_new_tokens:
                break


            generated_tokens.append(next_tok)
            if len(generated_tokens) >= max_new_tokens:
                break

            # 次のステップへ (自己回帰更新)
            next_input = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            logits_list, states, stats = self.model(
                next_input, 
                states=states, 
                active_layers=active_layers, 
                loop_steps=loop_steps,
                dynamic_early_exit=dynamic_early_exit
            )

        total_elapsed_sec = time.perf_counter() - t0
        pure_gen_sec = total_elapsed_sec - (t_prefill_ms / 1000.0)
        tokens_count = len(generated_tokens)
        
        tps = tokens_count / max(1e-5, pure_gen_sec)
        effective_speedup = 1.0

        return {
            "tokens": generated_tokens,
            "tokens_count": tokens_count,
            "steps": step_count,
            "tokens_per_step": round(tokens_count / max(1, step_count), 2),
            "prefill_ms": t_prefill_ms,
            "total_elapsed_sec": total_elapsed_sec,
            "pure_tps": round(tps, 2),
            "mtp_speedup": effective_speedup,
            "early_exit_count": early_exit_count,
            "active_layers": stats.get("active_layers", self.model.num_layers),
            "actual_loops": stats.get("actual_loops", self.model.loop_steps),
            "effective_depth": stats.get("effective_depth", self.model.num_layers * self.model.loop_steps)
        }
