# -*- coding: utf-8 -*-
"""分块Bootstrap置信区间: 对回测净值日收益率做块重采样, 估计总收益/回撤/夏普的分布。
用法: BT_JSON=output/backtest.json python tools/bootstrap_ci.py"""
import json, os
import numpy as np

res = json.load(open(os.environ.get("BT_JSON", "output/backtest.json")))
nav = np.array([e["nav"] for e in res["equity"]], dtype=float)
n = len(nav); rets = nav[1:] / nav[:-1] - 1
rng = np.random.default_rng(42)
B, L = int(os.environ.get("B", 3000)), 10
totals, mdds, sharpes = [], [], []
for _ in range(B):
    idx = rng.integers(0, len(rets) - L + 1, size=int(np.ceil(len(rets) / L)))
    sampled = np.concatenate([rets[i:i+L] for i in idx])[:len(rets)]
    curve = nav[0] * np.cumprod(1 + sampled)
    totals.append((curve[-1] / nav[0] - 1) * 100)
    mdds.append(float((curve / np.maximum.accumulate(curve) - 1).min()) * 100)
    sharpes.append(sampled.mean() / sampled.std() * np.sqrt(252) if sampled.std() else 0)
p = lambda a, q: round(float(np.percentile(a, q)), 2)
print(f"总收益  P5~P95: {p(totals,5)}% ~ {p(totals,95)}% (中位 {p(totals,50)}%)")
print(f"最大回撤 P5~P95: {p(mdds,5)}% ~ {p(mdds,95)}% (中位 {p(mdds,50)}%)")
print(f"夏普    P5~P95: {p(sharpes,5)} ~ {p(sharpes,95)} (中位 {p(sharpes,50)})")
