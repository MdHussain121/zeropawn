from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + res)
        return x


class Connect4Net(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_blocks: int = 3,
        num_filters: int = 64,
        num_actions: int = 7,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        # Stem
        self.stem_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(num_filters)

        # Residual backbone
        self.blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 16, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 6 * 7, num_actions)

        # Value head (WDL: Win, Draw, Loss)
        self.value_conv = nn.Conv2d(num_filters, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Stem
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        # Body
        for block in self.blocks:
            out = block(out)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        p_logits = self.policy_fc(p)

        # Value (WDL)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        wdl_logits = self.value_fc2(v)

        return p_logits, wdl_logits

    @torch.inference_mode()
    def predict_single(
        self, state_tensor: torch.Tensor, device: str | torch.device = "cpu"
    ) -> tuple[list[float], float]:
        """Predict policy probabilities and scalar search value P(win) - P(loss).

        Note: the caller is responsible for eval mode (call ``model.eval()``
        before invoking this); this method never toggles training mode.
        """
        dev = torch.device(device)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.to(dev)
        p_logits, wdl_logits = self(state_tensor)

        # Policy softmax
        p_probs = F.softmax(p_logits, dim=-1).squeeze(0).cpu().tolist()

        # WDL softmax: [P(win), P(draw), P(loss)]
        wdl_probs = F.softmax(wdl_logits, dim=-1).squeeze(0).cpu()
        # Search value = P(win) - P(loss)
        scalar_val = (wdl_probs[0] - wdl_probs[2]).item()

        return p_probs, scalar_val
