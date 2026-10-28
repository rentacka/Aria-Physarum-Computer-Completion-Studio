# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Studio 35: Aria Engraft DeltaLoop Model (core/aria_engraft_deltaloop.py)
================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【ENGRAFT 型アーキテクチャの真髄】
1. 本物 Gemma-2-2B の 256,000 語彙埋め込み (2304D) を 100% 凍結して再利用！
   -> 辞書をゼロから学ぶ必要が 0%！最初から日本語・英語・コードの意味空間が完全整列！
2. ENGRAFT 射影アダプタ (2304D -> 512D -> 2304D):
   -> Gemma の 2304次元高次元表現を、DeltaLoop の 512次元潜在思考空間へ低ランク射影！
3. 0-KV Gated DeltaNet 思考中枢 (Looped Transformer):
   -> 384KB 固定定常メモリ & 超高速 DynamicParameter ギアチェンジ！
4. 学習パラメータ数はわずか 27M (重みサイズ約 108 MB)！
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from safetensors import safe_open

from core.aria_deltaloop_model import LoopedDeltaBlock


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class AriaEngraftDeltaLoopModel(nn.Module):
    """
    🌸 👑 ⚡ Aria Engraft DeltaLoop Model
    - Gemma-2 2304D 埋め込み層 (凍結) ✕ ENGRAFT 双方向射影 ✕ 0-KV DeltaLoop
    """
    def __init__(
        self,
        gemma_embed_weight: torch.Tensor,
        d_model: int = 512,
        num_heads: int = 8,
        d_state: int = 32,
        num_layers: int = 12,
        loop_steps: int = 4
    ):
        super().__init__()
        vocab_size, d_gemma = gemma_embed_weight.shape
        self.vocab_size = vocab_size
        self.d_gemma = d_gemma
        self.d_model = d_model
        self.num_layers = num_layers
        self.loop_steps = loop_steps

        # 1. 本物 Gemma-2 埋め込み層 (凍結)
        self.embed_tokens = nn.Embedding.from_pretrained(gemma_embed_weight.float(), freeze=True)

        # 2. ENGRAFT 入力ダウンプロジェクション: 2304 -> 512
        self.in_proj = nn.Linear(d_gemma, d_model, bias=False)
        # 直交初期化
        nn.init.orthogonal_(self.in_proj.weight, gain=1.0)

        # 3. 0-KV Gated DeltaNet 思考中枢
        self.layers = nn.ModuleList([
            LoopedDeltaBlock(d_model=d_model, num_heads=num_heads, d_state=d_state)
            for _ in range(num_layers)
        ])

        # 4. ENGRAFT 出力アッププロジェクション: 512 -> 2304
        self.out_proj = nn.Linear(d_model, d_gemma, bias=False)
        nn.init.orthogonal_(self.out_proj.weight, gain=1.0)

        self.final_norm = RMSNorm(d_gemma, eps=1e-6)

        # 5. 生体周波数記憶 粘菌 Safetensors コア (ゼロ初期化ゲート結合)
        self.physarum_cortex = None
        self.physarum_gain = 0.0

    def attach_physarum_cortex(self, cortex, gain: float = 0.0):
        """粘菌 Safetensors 周波数コアの装着 (gain=0.0 で元の精度 100% 死守)"""
        self.physarum_cortex = cortex
        self.physarum_gain = float(gain)

    def set_physarum_gain(self, gain: float):
        """粘菌共鳴ゲインの動的切替 (0.0: 対話温存, 0.02~0.05: コード/推論オシレーション)"""
        self.physarum_gain = float(gain)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        active_layers: Optional[int] = None,
        loop_steps: Optional[int] = None,
        dynamic_early_exit: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        """
        Forward パス:
        Token IDs -> Gemma Embed (2304D) -> in_proj (512D) -> DeltaLoop -> out_proj (2304D) -> Tied LM Head (256000 Logits)
        """
        batch_size, seq_len = input_ids.shape
        num_active = active_layers if active_layers is not None else self.num_layers
        num_loops = loop_steps if loop_steps is not None else self.loop_steps

        # 1. 埋め込み & ダウンプロジェクション
        embeds = self.embed_tokens(input_ids) # [B, T, 2304]
        h = self.in_proj(embeds)              # [B, T, 512]

        # ★ 生体周波数共鳴: ゼロ初期化ゲート (gain=0.0 なら 100% 介入なし・精度完全温存)
        if self.physarum_cortex is not None and self.physarum_gain > 0.0:
            flux = self.physarum_cortex.compute_resonance_flux(h)
            h = h + self.physarum_gain * flux

        # 状態の初期化 (0-KV 定常メモリ)
        if states is None:
            states = [None] * num_active

        next_states = []
        actual_loops_run = 0
        exited_early = False

        # 2. 再帰ループ思考
        for loop in range(num_loops):
            actual_loops_run += 1
            h_prev = h
            for layer_idx in range(num_active):
                layer = self.layers[layer_idx]
                prev_state = states[layer_idx] if loop == 0 else next_states[layer_idx]
                h_out, new_state = layer(h, state=prev_state)
                h = h + h_out
                if loop == 0:
                    next_states.append(new_state)
                else:
                    next_states[layer_idx] = new_state

            # DynamicParameter アーリーイグジット判定
            if dynamic_early_exit and loop >= 1:
                diff_norm = (h - h_prev).norm() / (h.norm() + 1e-8)
                if diff_norm < 0.05:
                    exited_early = True
                    break

        # 3. アッププロジェクション & 正規化 (512D -> 2304D)
        h_gemma = self.out_proj(h)            # [B, T, 2304]
        h_normed = self.final_norm(h_gemma)   # [B, T, 2304]

        # 4. 本物 Gemma Tied-Weight によるロジット生成
        # logits = h_normed @ embed_tokens.weight.T
        logits = F.linear(h_normed, self.embed_tokens.weight) # [B, T, 256000]

        stats = {
            "active_layers": num_active,
            "actual_loops": actual_loops_run,
            "effective_depth": num_active * actual_loops_run,
            "exited_early": exited_early
        }

        return logits, next_states, stats

    @classmethod
    def load_from_gemma_cache(cls, cache_dir: str, d_model: int = 512, num_layers: int = 12, loop_steps: int = 4):
        """Gemma-2-2B の safetensors から埋め込み重みを自動抽出してモデルを構築"""
        file_path = os.path.join(cache_dir, "model-00001-of-00003.safetensors")
        with safe_open(file_path, framework="pt") as f:
            embed_weight = f.get_tensor("model.embed_tokens.weight")
        return cls(
            gemma_embed_weight=embed_weight,
            d_model=d_model,
            num_heads=8,
            d_state=32,
            num_layers=num_layers,
            loop_steps=loop_steps
        )

    @classmethod
    def load_from_gemma4_weights(cls, weight_path: str, d_model: int = 512, num_layers: int = 12, loop_steps: int = 4):
        """Gemma-4-E4B-it の ENGRAFT 重み (gemma4_engraft_weights.pt) からモデルを構築"""
        data = torch.load(weight_path, map_location="cpu", weights_only=True)
        embed_weight = data["embed_tokens"] # [262144, 2560]
        model = cls(
            gemma_embed_weight=embed_weight,
            d_model=d_model,
            num_heads=8,
            d_state=32,
            num_layers=num_layers,
            loop_steps=loop_steps
        )
        if data.get("norm") is not None:
            norm_weight = data["norm"].float()
            model.final_norm.weight.data.copy_(norm_weight)
        return model

