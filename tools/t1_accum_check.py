"""T1 梯度累积的正确性自检：组算术 + 归一化梯度等价性。

两件事：
1. 组算术：任意 n_micro / accum，分组必须**每个微批恰好属于一组**、每组都会触发一次
   优化器步、每轮总优化器步数 = ceil(n_micro/accum)，且最后一组即使不足也能收尾。
2. 归一化：把一批 12 个样本拆成 3 个微批累积，梯度应等于整批 12 个样本平均损失的梯度
   （前提：微批等大、loss 用 mean）。这是协议里"损失按实际累计样本数归一化"的数学依据。

    python tools/t1_accum_check.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def group_plan(n_micro: int, accum: int):
    """复刻 cxr/train.py 训练循环里的分组算术。"""
    plan = []
    for idx in range(n_micro):
        g_start = (idx // accum) * accum
        g_micro = min(accum, n_micro - g_start)
        plan.append((g_micro, ((idx % accum) + 1) >= g_micro))
    return plan


def check_groups():
    bad = 0
    for accum in (1, 2, 3, 4, 8):
        for n_micro in (1, 2, 3, 5, 7, 11, 1511, 1512, 1513):
            plan = group_plan(n_micro, accum)
            steps = sum(1 for _, step in plan if step)
            expect = math.ceil(n_micro / accum)
            # 每个微批都必须被"覆盖"到一次 step（最后一组的收尾算在内）
            ok_steps = steps == expect
            # 组内实际微批数之和必须等于总微批数（只在每组收尾时累加一次）
            ok_sum = sum(g for g, step in plan if step) == n_micro
            # 最后一个微批必须是优化器步（否则梯度会跨 epoch 泄漏）
            ok_tail = plan[-1][1]
            if not (ok_steps and ok_sum and ok_tail):
                bad += 1
                print(f"  FAIL accum={accum} n_micro={n_micro}: steps={steps} expect={expect} "
                      f"sum={ok_sum} tail={ok_tail}")
    print(f"[check] 组算术: {'全部通过' if bad == 0 else f'{bad} 个失败'}")
    return bad == 0


def check_gradient_equivalence():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 10))
    x = torch.randn(12, 8)
    y = (torch.rand(12, 10) > 0.5).float()
    lossf = nn.BCEWithLogitsLoss()

    # --- 整批一次算 ---
    net.zero_grad(set_to_none=True)
    lossf(net(x), y).backward()
    g_full = [p.grad.detach().clone() for p in net.parameters()]

    # --- 拆成 3 个微批，按 1/3 归一化累积 ---
    net.zero_grad(set_to_none=True)
    g_micro = 3
    for i in range(3):
        s = slice(i * 4, (i + 1) * 4)
        (lossf(net(x[s]), y[s]) * (1.0 / g_micro)).backward()
    g_acc = [p.grad.detach().clone() for p in net.parameters()]

    dmax = max((a - b).abs().max().item() for a, b in zip(g_full, g_acc))
    rel = max(((a - b).abs().max() / (b.abs().max() + 1e-12)).item()
              for a, b in zip(g_full, g_acc))
    print(f"[check] 梯度等价: max|Δ| = {dmax:.3e}  max rel = {rel:.3e}  "
          f"({'通过' if dmax < 1e-6 else '失败'})")
    return dmax < 1e-6


if __name__ == "__main__":
    ok = check_groups() and check_gradient_equivalence()
    raise SystemExit(0 if ok else 1)
