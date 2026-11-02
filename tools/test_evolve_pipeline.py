#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evolve 管道全链路自检（第一百二十二轮）。

为什么需要它：首跑的两个候选都死在第一道门（G1），G3 分年、G4 邻域、G5 封存区
这三段代码**从未被执行过**。而它们恰恰是唯一会消耗封存区预算、并触发自动改写
config 的路径 —— 用真实回测验证一次要 20 分钟且必须消耗预算，用假回测验证则
可以反复跑、且能构造出"恰好通过"和"恰好卡在分年"两种边界。

做法：给 `src.backtest` 打桩（注入假的 prepare_context / run_backtest），
把 evolve 的 ROOT/EVO 指向临时目录（不污染真账本、不消耗真预算），
然后真的调用 cmd_run / cmd_apply，检查账本、state、报告、config 改写。
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT_REAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_REAL))
sys.path.insert(0, str(ROOT_REAL / "tools"))

import evolve  # noqa: E402

# ── 桩：假行情与假回测 ──────────────────────────────────────────────
DATES = [f"{y}{m:02d}{d:02d}" for y in range(2016, 2026) for m, d in ((3, 1), (6, 1), (9, 1), (12, 1))]
DATES += ["20250701", "20250901", "20251101", "20260101", "20260301", "20260501"]


class FakeBacktest:
    """按 config 里的探针值返回不同的**逐年**收益曲线，用来精确命中每一道门。

    基线与候选的差距必须是"分年形状"而不是"总数"，否则分年闸门永远测不到：
      991 全门通过（7 年改善）
      992 分年不过（4/10 改善，总数更高 —— 证明 G3 能拦住"总数好看但结构集中"）
      993 候选本身通过，但网格档位卡玛掉 0.5 —— 证明 G4 能拦住孤立尖峰
      994 网格档位（卡玛 4.5）
    """

    BASE_YEARS = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    PROFILES = {
        0: (BASE_YEARS, -20.0, 5.00, 300),
        991: ([25, 25, 25, 25, 25, 25, 25, 16, 16, 16], -19.5, 5.50, 310),
        992: ([15, 15, 15, 15, 15, 15, 40, 40, 40, 40], -19.5, 5.50, 310),
        993: ([25, 25, 25, 25, 25, 25, 25, 16, 16, 16], -19.5, 5.20, 310),
        994: (BASE_YEARS, -20.0, 4.50, 300),
    }

    @staticmethod
    def prepare_context(cfg, hist_days=None):
        return {"main_dates": list(DATES)}

    @classmethod
    def run_backtest(cls, cfg, _unused, ctx):
        probe = int(cfg["execution"].get("sell_close_if_open_drop", 0) or 0)
        prof = cls.PROFILES.get(probe) or cls.PROFILES[0]
        years_ret, mdd, karma, trades = prof
        dates = ctx["main_dates"]
        years = sorted({d[:4] for d in dates})
        eq, nav = [], 1.0
        for i, y in enumerate(years):
            ys = [d for d in dates if d[:4] == y]
            if not ys:
                continue
            r = years_ret[min(i, len(years_ret) - 1)] / 100
            eq.append({"d": ys[0], "nav": nav})
            nav *= (1 + r)
            eq.append({"d": ys[-1], "nav": nav})
        return {"metrics": {"总收益率%": (nav - 1) * 100, "最大回撤%": mdd, "卡玛比率": karma,
                            "交易次数": trades, "胜率%": 55.0, "平均单笔收益%": 1.0},
                "equity": eq}


def setup_tmp() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="evo_selftest_"))
    (tmp / "evolution").mkdir()
    (tmp / "evolution" / "reports").mkdir()
    shutil.copy(ROOT_REAL / "config.yaml", tmp / "config.yaml")
    for f in ("policy.yaml",):
        shutil.copy(ROOT_REAL / "evolution" / f, tmp / "evolution" / f)
    (tmp / "evolution" / "candidates.yaml").write_text(
        "- id: ok_struct\n"
        "  kind: structure\n"
        '  expires: "2027-12-31"\n'
        "  patch: {execution.sell_close_if_open_drop: 991}\n"
        '  question: "自检：全门通过"\n'
        "  grid: {execution.sell_close_if_open_drop: [0, 991]}\n"
        "- id: bad_years\n"
        "  kind: structure\n"
        '  expires: "2027-12-31"\n'
        "  patch: {execution.sell_close_if_open_drop: 992}\n"
        '  question: "自检：分年崩塌"\n'
        "- id: bad_neighbor\n"
        "  kind: structure\n"
        '  expires: "2027-12-31"\n'
        "  patch: {execution.sell_close_if_open_drop: 993}\n"
        '  question: "自检：孤立尖峰"\n'
        "  grid: {execution.sell_close_if_open_drop: [994, 994]}\n", encoding="utf-8")
    return tmp


def point_evolve_at(tmp: Path):
    evolve.ROOT = tmp
    evolve.EVO = tmp / "evolution"
    evolve.POLICY_F = evolve.EVO / "policy.yaml"
    evolve.CANDS_F = evolve.EVO / "candidates.yaml"
    evolve.SEED_F = evolve.EVO / "history_seed.yaml"
    evolve.LEDGER_F = evolve.EVO / "ledger.jsonl"
    evolve.STATE_F = evolve.EVO / "state.json"
    evolve.LAST_F = evolve.EVO / "last_run.json"
    evolve.LAST_MD = evolve.EVO / "last_run.md"
    evolve.REPORTS = evolve.EVO / "reports"


def main() -> int:
    sys.modules["src.backtest"] = FakeBacktest  # type: ignore[assignment]
    import src  # noqa: F401  确保 src 包存在，再覆盖子模块
    sys.modules["src.backtest"] = FakeBacktest  # type: ignore[assignment]

    tmp = setup_tmp()
    point_evolve_at(tmp)
    ok = True

    # ── 第一轮：三个候选各命中一条不同的门
    evolve.cmd_run(argparse.Namespace(cmd="run", max=3, only="", smoke=False, dry_run=False))
    led = [json.loads(x) for x in (tmp / "evolution" / "ledger.jsonl").read_text().splitlines() if x]
    dec = {r["candidate_id"]: r["decision"] for r in led}
    st = json.loads((tmp / "evolution" / "state.json").read_text())

    expect = {"ok_struct": "PASS", "bad_years": "REJECT_YEARS", "bad_neighbor": "REJECT_SPIKE"}
    for cid, want in expect.items():
        got = dec.get(cid)
        good = got == want
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {cid:14s} 期望 {want:14s} 实际 {got}")

    # 预算：只有 ok_struct 走到了 G5，其余两个在 G5 之前就被否
    year = str(__import__("datetime").datetime.now().year)
    used = (st.get("holdout_used") or {}).get(year)
    budget_ok = used == 1
    ok &= budget_ok
    print(f"[{'PASS' if budget_ok else 'FAIL'}] 封存区预算计账 {st.get('holdout_used')}（期望 {year}=1）")

    # 账本只写真裁决，不写 SKIP
    no_skip = all(not r["decision"].startswith("SKIP") for r in led)
    ok &= no_skip
    print(f"[{'PASS' if no_skip else 'FAIL'}] 账本 {len(led)} 条，无 SKIP_* 记录")

    # 报告归档 + windows 元数据
    rep = sorted((tmp / "evolution" / "reports").glob("*.md"))
    last = json.loads((tmp / "evolution" / "last_run.json").read_text())
    rep_ok = len(rep) == 1 and last.get("windows", {}).get("fast", {}).get("days")
    ok &= bool(rep_ok)
    print(f"[{'PASS' if rep_ok else 'FAIL'}] 报告归档 {len(rep)} 份，windows={list(last.get('windows') or {})}")

    # ── 第二轮：重跑必须被账本去重（PASS → SKIP_APPLIED）
    evolve.cmd_run(argparse.Namespace(cmd="run", max=0, only="ok_struct", smoke=False, dry_run=False))
    last2 = json.loads((tmp / "evolution" / "last_run.json").read_text())
    d2 = {v["candidate_id"]: v["decision"] for v in last2["verdicts"]}
    dedup_ok = d2.get("ok_struct") == "SKIP_APPLIED"
    ok &= dedup_ok
    print(f"[{'PASS' if dedup_ok else 'FAIL'}] 重跑去重 {d2}")

    # ── 第三轮：apply 必须能改 config 且不丢注释
    before = (tmp / "config.yaml").read_text()
    evolve.cmd_apply(argparse.Namespace(id="ok_struct", force=False))
    after = (tmp / "config.yaml").read_text()
    import yaml
    cfg2 = yaml.safe_load(after)
    body = after.split("\n", 1)[1] if after.startswith("# [EVOLVE]") else after
    applied_ok = (cfg2["execution"]["sell_close_if_open_drop"] == 991
                  and body.count("#") == before.count("#")
                  and after.startswith("# [EVOLVE]"))
    ok &= applied_ok
    st2 = json.loads((tmp / "evolution" / "state.json").read_text())
    print(f"[{'PASS' if applied_ok else 'FAIL'}] apply 生效，策略版本 {st2.get('strategy_version')}，"
          f"注释行 {before.count('#')}→{body.count('#')}")

    # 重复 apply 必须幂等（第二次直接跳过）
    rc = evolve.cmd_apply(argparse.Namespace(id="ok_struct", force=False))
    idem_ok = rc == 0 and (tmp / "config.yaml").read_text() == after
    ok &= idem_ok
    print(f"[{'PASS' if idem_ok else 'FAIL'}] apply 重复调用幂等（rc={rc}）")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("全部通过 ✅" if ok else "存在失败项 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
