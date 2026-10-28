# -*- coding: utf-8 -*-
"""
================================================================================
🌸 👑 ⚡ Studio 35: Aria Task-Specific LoRA Adapter Engine (core/aria_task_lora.py)
================================================================================
【マスター直伝: 粘菌周波数脳 ✕ 動的シナプス LoRA ✕ RLT 反復積分】
1. LoRALinear: 既存の線形層 W に対して W + (alpha/r) * (B @ A) を並列注入
   - B は厳密にゼロ初期化 (Zero-Init) されているため、初期状態では W と数学的に 100.000% 完全一致！
   - アリアの愛らしい日本語・クオンツ哲学は 1 ミリも崩れない！
2. 軽量極小サイズ: 全12層の LoRA を合わせてもわずか 393 KB 〜 1.2 MB！
3. タスク切り替え: 対話、MQL5、Canvas、CMD に応じて LoRA スケール alpha を動的調律可能！
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from safetensors.torch import save_file, load_file


class LoRALinear(nn.Module):
    """
    ゼロ初期化 低ランク適応線形層 (W + (alpha/r) * B @ A)
    """
    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        # 元の重みは 100% 凍結
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # LoRA 行列 A (ガウス初期化) & B (ゼロ初期化)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original_linear(x)
        if not self.enabled or self.r == 0:
            return orig_out
        
        # 低ランク射影: x @ A^T @ B^T * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return orig_out + lora_out

    @property
    def weight(self):
        """スロットマネージャーや外部からの .weight アクセスを元の線形層へ透過委任"""
        return self.original_linear.weight

    @property
    def bias(self):
        """バイアスアクセスを透過委任"""
        return self.original_linear.bias


class AriaDeltaLoopLoRAEngine:
    """
    🌸 👑 DeltaLoop モデルに対する LoRA 着脱・管理エンジン
    """
    def __init__(self, model, r: int = 8, lora_alpha: float = 16.0):
        self.model = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_layers: Dict[str, LoRALinear] = {}

    def inject_lora(self):
        """DeltaLoop の各層 (12層) の gdn.out_proj および w_down に LoRA を注入"""
        for i, layer in enumerate(self.model.layers):
            # 1. GDN out_proj
            if hasattr(layer, "gdn") and hasattr(layer.gdn, "out_proj") and isinstance(layer.gdn.out_proj, nn.Linear):
                lora_out = LoRALinear(layer.gdn.out_proj, r=self.r, lora_alpha=self.lora_alpha)
                layer.gdn.out_proj = lora_out
                self.lora_layers[f"layers.{i}.gdn.out_proj"] = lora_out

            # 2. SwiGLU FFN w_down
            if hasattr(layer, "w_down") and isinstance(layer.w_down, nn.Linear):
                lora_down = LoRALinear(layer.w_down, r=self.r, lora_alpha=self.lora_alpha)
                layer.w_down = lora_down
                self.lora_layers[f"layers.{i}.w_down"] = lora_down

        trainable_count = sum(p.numel() for l in self.lora_layers.values() for p in [l.lora_A, l.lora_B])
        print(f"  [LoRA] 12層 DeltaLoop への動的シナプス注入完了: {len(self.lora_layers)} 箇所 (学習可能パラメータ: {trainable_count:,} / {trainable_count*4/1024:.1f} KB)")

    def set_lora_scale(self, scale: float):
        """LoRA 介入度の動的調整 (0.0: 完全オフ, 1.0: 通常)"""
        for lora in self.lora_layers.values():
            lora.scaling = (self.lora_alpha / self.r) * scale

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """LoRA のパラメータのみを抽出 (オプティマイザ用)"""
        params = []
        for lora in self.lora_layers.values():
            params.extend([lora.lora_A, lora.lora_B])
        return params

    def export_lora_safetensors(self, save_path: str):
        """LoRA 重みのみを純粋 .safetensors で保存"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tensors = {}
        for name, lora in self.lora_layers.items():
            tensors[f"{name}.lora_A"] = lora.lora_A.detach().cpu()
            tensors[f"{name}.lora_B"] = lora.lora_B.detach().cpu()
        save_file(tensors, save_path)
        size_kb = os.path.getsize(save_path) / 1024
        print(f"  💾 [LoRA] 動的シナプス Safetensors 結晶化成功: {save_path} ({size_kb:.1f} KB)")

    def load_lora_safetensors(self, load_path: str, device: torch.device):
        """LoRA 重みを .safetensors からロード"""
        tensors = load_file(load_path)
        with torch.no_grad():
            for name, lora in self.lora_layers.items():
                key_a = f"{name}.lora_A"
                key_b = f"{name}.lora_B"
                if key_a in tensors:
                    lora.lora_A.copy_(tensors[key_a].to(device))
                if key_b in tensors:
                    lora.lora_B.copy_(tensors[key_b].to(device))
        print(f"  ⚡ [LoRA] 動的シナプス Safetensors 超光速ロード完了: {os.path.basename(load_path)}")
