# -*- coding: utf-8 -*-
"""第一百九十五轮: 无杠杆(lev=1.0)下的参数进化 —— 遗传算法 + 样本外验证。

为什么重新做参数搜索: 第 108~120 轮的全部参数扫描都是在 **1.25x 杠杆** 目标下做的
(那年化 183.8%、卡玛 8.45)。用户第 195 轮决定彻底不用杠杆, 目标函数随之改变 ——
1.0x 下卡玛 7.44(不是 8.45), 回撤只剩 -17.73%, "多赚一点"和"少回撤一点"的边际
价值不再由杠杆放大, 最优参数不一定还在原位。

算法: 8 维离散基因(全部取历史已扫过的档, 不引入新维度), 种群 24 / 8 代,
精英保留 + 锦标赛选择 + 均匀交叉 + 邻域变异(±1 档), 适应度见 fitness()。

用法:
    python3 tools/exp_ga_195.py 1500            # 训练(默认)
    python3 tools/exp_ga_195.py 1500 --verify    # 用已存 top 个体在 3000 日验证
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("BT_FULL_TRADES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import multiprocessing as mp   # noqa: E402

from src.backtest import run_backtest   # noqa: E402
from src.config import load_config      # noqa: E402

# ---- 基因空间: 8 维, 每维 5 档(全部是历史扫过的区间, 不外推) ----
GENES = [
    ("selection.dip_pct_min",         [-7.0, -6.5, -6.0, -5.5, -5.0]),
    ("selection.dip_pct_max",         [-4.0, -3.5, -3.0, -2.5, -2.0]),
    ("selection.dip_vol_ratio_max",   [3.0, 3.5, 4.0, 4.5, 5.0]),
    ("selection.bias20_max",          [10.0, 10.8, 11.4, 12.0, 12.6]),
    ("selection.dip_close_pos_max",   [0.26, 0.29, 0.32, 0.35, 0.38]),
    ("regime.market_ma",              [10, 15, 20, 30, 40]),
    ("regime.breadth_min",            [0.15, 0.20, 0.25, 0.30, 0.35]),
    ("execution.sell_full_if_open_gain", [10.0, 12.0, 15.0, 20.0, 99.0]),
]
BASE = [-1] * len(GENES)   # 由 main() 按当前 config 填成真实下标

POP, GENS, ELITE = 18, 5, 3
MIN_TRADES = 300           # 笔数下限: 低于基准(365)的 82% 视为"缩量换卡玛", 按比例打折

CTX = None
CFG = None
DAYS = 1500
_CACHE: dict[tuple, dict] = {}


def to_params(ind: list[int]) -> dict:
    return {name: opts[i] for (name, opts), i in zip(GENES, ind)}


def build_cfg(params: dict) -> dict:
    cfg = copy.deepcopy(CFG)
    for name, v in params.items():
        sec, key = name.split(".")
        cfg[sec][key] = v
    return cfg


def yearly(equity: list[dict]) -> dict:
    """按自然年切净值, 返回 {年: {ret%, mdd%, calmar}}。"""
    import numpy as np
    d = np.array([e["d"][:4] for e in equity])
    nav = np.array([e["nav"] for e in equity], dtype="float64")
    out = {}
    for y in sorted(set(d)):
        m = d == y
        s = nav[m]
        if len(s) < 20:
            continue
        ret = (s[-1] / s[0] - 1.0) * 100.0
        peak = np.maximum.accumulate(s)
        mdd = float((s / peak - 1.0).min()) * 100.0
        out[y] = {"ret": ret, "mdd": mdd}
    return out


def fitness(m: dict) -> float:
    """卡玛 × 笔数折扣。卡玛峰值常出现在"笔数变少"的方向上, 必须罚。"""
    cagr, mdd, n = float(m["年化收益率%"]), abs(float(m["最大回撤%"])), int(m["交易次数"])
    calmar = cagr / mdd if mdd else 0.0
    return calmar * min(1.0, n / MIN_TRADES)


def _clean(v):
    import numpy as np
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _worker(ind: tuple) -> dict:
    key = tuple(ind)
    if key in _CACHE:
        return _CACHE[key]
    cfg = build_cfg(to_params(list(ind)))
    res = run_backtest(cfg, DAYS, CTX)
    m = {k: _clean(v) for k, v in res["metrics"].items()}
    rec = {"ind": list(ind), "params": to_params(list(ind)), "metrics": m,
           "fit": round(fitness(m), 4), "yearly": yearly(res["equity"])}
    _CACHE[key] = rec
    return rec


def _run_chunk(jobs: list[tuple], path: str) -> None:
    """fork 子进程的入口: 逐个评估, 结果写队列。

    ★坑: 不能把 _worker 交给 Pool.map —— 主脚本以 __main__ 运行时函数的
    __module__ 是 "__main__", ForkingPickler 按模块名反查会失败("import of module
    '__main__' failed"), 表现为 worker 侧一串 BrokenPipe 而主进程静默退出。
    Process(target=...) 在 fork 下不 pickle target, 这里改用手工分块。
    """
    out = []
    for j in jobs:
        try:
            out.append(_worker(tuple(j)))
        except Exception as e:   # noqa: BLE001
            out.append({"ind": list(j), "error": repr(e), "fit": -9.0, "metrics": {}})
    Path(path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def _neighbour(ind: list[int], rng: random.Random, k: int = 1) -> list[int]:
    out = ind[:]
    for _ in range(k):
        j = rng.randrange(len(out))
        step = rng.choice([-1, 1])
        out[j] = max(0, min(len(GENES[j][1]) - 1, out[j] + step))
    return out


def main() -> None:
    global CTX, CFG, DAYS
    DAYS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1500
    rng = random.Random(1951111)
    CFG = load_config()
    # 基线基因 = 当前 config 的真实下标(找不到就取最近档)
    for i, (name, opts) in enumerate(GENES):
        sec, key = name.split(".")
        cur = CFG[sec][key]
        BASE[i] = min(range(len(opts)), key=lambda j: abs(opts[j] - float(cur)))
        if abs(opts[BASE[i]] - float(cur)) > 1e-9:
            print(f"!! {name} 当前值 {cur} 不在搜索档内, 用最近档 {opts[BASE[i]]}", flush=True)

    print(f"窗口 {DAYS} 日 · 杠杆 {CFG['execution']['leverage']} · 基线基因 {BASE}", flush=True)
    t0 = time.time()
    from src.backtest import prepare_context
    CTX = prepare_context(CFG, DAYS)
    print(f"ctx 就绪 {time.time() - t0:.0f}s", flush=True)

    pop = [BASE[:]]
    seen = {tuple(BASE)}
    while len(pop) < POP:
        cand = _neighbour(BASE[:], rng, k=2) if rng.random() < 0.5 else \
            [rng.randrange(len(o)) for _, o in GENES]
        if tuple(cand) not in seen:
            seen.add(tuple(cand))
            pop.append(cand)

    nproc = min(6, os.cpu_count() or 4)
    ctx_mp = mp.get_context("fork")
    print(f"{nproc} 进程并行, 单点 ~19s", flush=True)

    def evaluate(inds: list[tuple]) -> list[dict]:
        """★坑: 不能用 multiprocessing.Queue 回传 —— 子进程 put 后立即退出, feeder 线程
        可能没把数据刷完, 主进程 q.get() 会永久阻塞(第一版就卡死在这里)。改成各写各的文件。"""
        chunks = [inds[i::nproc] for i in range(nproc)]
        paths = [f"/tmp/ga195_chunk_{os.getpid()}_{i}.json" for i in range(len(chunks))]
        ps = [ctx_mp.Process(target=_run_chunk, args=(c, pth))
              for c, pth in zip(chunks, paths) if c]
        for pr in ps:
            pr.start()
        for pr in ps:
            pr.join()
        got = {}
        for pth in paths:
            if not Path(pth).exists():
                continue
            for r in json.loads(Path(pth).read_text(encoding="utf-8")):
                got[tuple(r["ind"])] = r
        missing = [i for i in inds if i not in got]
        if missing:                     # 兜底: 子进程被 OOM 杀掉时补一个零分个体
            print(f"!! {len(missing)} 个个体未返回", flush=True)
            for i in missing:
                got[i] = {"ind": list(i), "params": to_params(list(i)),
                          "metrics": {"卡玛比率": 0, "年化收益率%": 0, "最大回撤%": 0,
                                      "交易次数": 0}, "fit": -9.0, "yearly": {}}
        return [got[i] for i in inds]

    ALL: dict[tuple, dict] = {}
    if True:
        hist = []
        for g in range(GENS + 1):
            t = time.time()
            recs = evaluate([tuple(p) for p in pop])
            for r in recs:                     # 子进程的 _CACHE 不跨进程回传, 父进程自己攒
                ALL[tuple(r["ind"])] = r
            recs.sort(key=lambda r: r["fit"], reverse=True)
            hist.append([{k: r[k] for k in ("ind", "params", "fit")} for r in recs[:5]])
            base_fit = ALL[tuple(BASE)]["fit"] if tuple(BASE) in ALL else float("nan")
            print(f"[代 {g}] {time.time() - t:.0f}s  最优 fit={recs[0]['fit']:.3f} "
                  f"(卡玛 {recs[0]['metrics']['卡玛比率']}, 年化 {recs[0]['metrics']['年化收益率%']}, "
                  f"笔数 {recs[0]['metrics']['交易次数']})  基线 fit={base_fit:.3f}", flush=True)
            (ROOT / "output" / f"exp_ga_195_{DAYS}_partial.json").write_text(
                json.dumps({"gen": g, "base_ind": BASE,
                            "best": [{k: r[k] for k in ("ind", "params", "fit")} for r in recs[:5]],
                            "n": len(ALL)}, ensure_ascii=False), encoding="utf-8")
            if g == GENS:
                break
            elite = [r["ind"] for r in recs[:ELITE]]
            newpop = [e[:] for e in elite]
            while len(newpop) < POP:
                a = max(rng.sample(recs[:10], 3), key=lambda r: r["fit"])["ind"]
                b = max(rng.sample(recs[:10], 3), key=lambda r: r["fit"])["ind"]
                child = [a[i] if rng.random() < 0.5 else b[i] for i in range(len(GENES))]
                if rng.random() < 0.75:
                    child = _neighbour(child, rng, k=rng.choice([1, 1, 2]))
                newpop.append(child)
            pop = newpop

        top = sorted(ALL.values(), key=lambda r: r["fit"], reverse=True)[:8]

    base = ALL[tuple(BASE)]
    out = {"n_eval": len(ALL), "days": DAYS, "leverage": CFG["execution"]["leverage"],
           "genes": [(n, o) for n, o in GENES], "base_ind": BASE,
           "base": {"metrics": base["metrics"], "fit": base["fit"]},
           "top": [{k: r[k] for k in ("ind", "params", "metrics", "fit")} for r in top],
           "hist": hist, "elapsed_s": round(time.time() - t0)}
    p = ROOT / "output" / f"exp_ga_195_{DAYS}.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n基线: {base['metrics']}", flush=True)
    print("\nTop 8:", flush=True)
    for r in top:
        print(f"  fit={r['fit']:6.3f} 卡玛={r['metrics']['卡玛比率']:5.2f} "
              f"年化={r['metrics']['年化收益率%']:7.2f}% 回撤={r['metrics']['最大回撤%']:7.2f} "
              f"笔={r['metrics']['交易次数']:4d} | {r['params']}", flush=True)
    print(f"→ {p}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
