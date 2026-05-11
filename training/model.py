"""
Dual-perspective NNUE: 768 → 256 (FT, shared) → 512 concat → 32 → 32 → 1

Both White and Black accumulators use the same FT weights (weight sharing).
The stm (side-to-move) accumulator is concatenated first:
    input = cat([stm_acc, opp_acc])

Quantisation targets (export.py):
    FT weights / biases :  ×QA=127  → i16
    L1/L2/out weights   :  ×QB=64   → i8
    L1 biases           :  ×QA×QB   → i32
    L2 / out biases     :  ×QB²     → i32
"""

import torch
import torch.nn as nn

INPUT_SIZE = 768   # 6 × 2 × 64
L1_SIZE    = 256   # FT output per perspective
L2_SIZE    = 32
L3_SIZE    = 32

QA       = 127.0
QB       = 64.0
SCALE_CP = 400.0   # sigmoid target = sigmoid(cp / SCALE_CP)  — win probability


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.Linear(INPUT_SIZE, L1_SIZE)
        self.l1 = nn.Linear(L1_SIZE * 2, L2_SIZE)
        self.l2 = nn.Linear(L2_SIZE, L3_SIZE)
        self.out = nn.Linear(L3_SIZE, 1)

        nn.init.uniform_(self.ft.weight, -0.01, 0.01)
        nn.init.zeros_(self.ft.bias)

    def forward(self, white_feat: torch.Tensor, black_feat: torch.Tensor,
                stm: torch.Tensor) -> torch.Tensor:
        """
        white_feat : [B, 768]  binary features from White's POV
        black_feat : [B, 768]  binary features from Black's POV
        stm        : [B]       int64  (0 = White to move, 1 = Black to move)
        returns    : [B, 1]    logit (pre-sigmoid) from stm's perspective
        """
        white_acc = torch.clamp(self.ft(white_feat), 0.0, 1.0)  # [B, 256]
        black_acc = torch.clamp(self.ft(black_feat), 0.0, 1.0)  # [B, 256]

        # stm first, opponent second — matches Rust Accumulator::evaluate order.
        is_black  = stm.bool().unsqueeze(1)                      # [B, 1]
        stm_acc   = torch.where(is_black, black_acc, white_acc)
        opp_acc   = torch.where(is_black, white_acc, black_acc)

        x = torch.cat([stm_acc, opp_acc], dim=1)   # [B, 512]
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        return self.out(x)                          # [B, 1]

    def forward_from_accumulators(self, stm_acc: torch.Tensor,
                                  opp_acc: torch.Tensor) -> torch.Tensor:
        """Run L1→output from pre-computed CReLU activations (testing / export)."""
        x = torch.cat([stm_acc, opp_acc], dim=-1)
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        return self.out(x)
