"""Dual-head Residual Network for ZeroPawn.

Contract Specifications:
- Input: 8x8x19 planes from side-to-move's perspective
- Network: conv stem + 12 residual blocks, 128 filters
- Policy head: 4672 outputs (8x8x73 AlphaZero encoding)
- Value head: 3 outputs (WDL: Win, Draw, Loss softmax)
- Parameters: 5M to 10M
- Autocast: bfloat16 compatible, channels-last memory format
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic 2-convolution residual block with Batch Normalization."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res, inplace=True)
        return out


class ChessNet(nn.Module):
    """Dual-head ResNet matching the Project Contract specification."""

    def __init__(
        self,
        in_channels: int = 19,
        num_filters: int = 128,
        num_blocks: int = 12,
        num_actions: int = 4672,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_filters = num_filters
        self.num_blocks = num_blocks

        # 1. Conv Stem
        self.stem_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(num_filters)

        # 2. Residual Backbone (12 blocks)
        self.blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_blocks)])

        # 3. Policy Head (conv 1x1 to 16 channels, then linear to 4672 actions)
        self.policy_conv = nn.Conv2d(num_filters, 16, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 8 * 8, num_actions)

        # 4. Value Head (WDL: 3 outputs for Win, Draw, Loss)
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 3)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 19, 8, 8)
            legal_mask: Optional boolean tensor of shape (B, 4672) where True = legal.

        Returns:
            policy_logits: (B, 4672)
            wdl_logits: (B, 3)
        """
        # Conv stem
        out = F.relu(self.stem_bn(self.stem_conv(x)), inplace=True)

        # ResNet body
        for block in self.blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)), inplace=True)
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        if legal_mask is not None:
            # Mask illegal moves with large negative value before softmax
            if legal_mask.dtype is not torch.bool:
                raise ValueError(f"legal_mask must have dtype torch.bool, got {legal_mask.dtype}")
            if legal_mask.shape != policy_logits.shape:
                raise ValueError(
                    f"legal_mask shape {tuple(legal_mask.shape)} must match "
                    f"policy_logits shape {tuple(policy_logits.shape)}"
                )
            fill_val = -1e4 if policy_logits.dtype in (torch.bfloat16, torch.float16) else -1e9
            policy_logits = policy_logits.masked_fill(~legal_mask, fill_val)

        # Value head (WDL)
        v = F.relu(self.value_bn(self.value_conv(out)), inplace=True)
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v), inplace=True)
        wdl_logits = self.value_fc2(v)

        return policy_logits, wdl_logits

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar search value = P(win) - P(loss) from WDL logits."""
        _, wdl_logits = self.forward(x)
        probs = F.softmax(wdl_logits, dim=-1)
        return probs[:, 0] - probs[:, 2]
