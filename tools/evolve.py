#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略自进化引擎 —— 由 GitHub Actions 周频驱动，只做"假设裁决"，不做"参数搜索"。

为什么要这样设计（第一百二十二轮）
------------------------------------
第一百二十一轮的实证结论是：**参数层已经不可辨识**。
  - `bias20_max` 5 档在 13 个年份上的 Kendall W = 0.058（chi2=3.03, df=4, 临界 9.49）
  - `dip_vol_ratio_max` W = 0.008，13 年里有 4 年五档结果完全相同
  - 随机 30 组参数 30/30 正收益，当前参数只是分布外的一个点（3.21x 中位）
在参数排名与抛硬币无异的空间里做自动搜索，产出的一定是过拟合。
所以本引擎把"进化"定义成三件不同的事：

  ① **结构进化**（本文件负责）：某个机制该不该存在。候选必须人工预注册，
     引擎只负责用三段信息隔离的闸门去证伪它。
  ② **风控进化**：由 `src/monitor.py` 的滚动失效探针负责，只告警不改仓，
     硬治理需单独验证（该模块的设计注释里已写明这条纪律）。
  ③ **参数搜索**：被 policy.autonomy.parameter = never 从治理层禁掉。

三段信息隔离
------------
    fast      封存区之前最近 1500 日   → 只做快筛，把明显坏的挡在昂贵回测之外
    extended  封存区之前全部历史       → 主口径，收益/卡玛/回撤/分年一致性
    holdout   封存区（默认 2025-07-01 起）→ 预算制，每年最多 4 次，用一次记一次

封存区不是"干净的样本外"（现行 baseline 当年也见过它），但它的污染方向是**不利于候选**的：
incumbent 在封存区上有选择优势，候选要在这里赢，只能靠增量。所以它是一道保守的门。

命令
----
    python tools/evolve.py seed                    # 回填历史裁决（幂等）
    python tools/evolve.py run [--max 2] [--only ID] [--smoke]
    python tools/evolve.py apply --id ID           # 把 PASS 记录落到 config.yaml（保注释）
    python tools/evolve.py state                   # 重写 evolution/STATE.md
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVO = ROOT / "evolution"
POLICY_F = EVO / "policy.yaml"
CANDS_F = EVO / "candidates.yaml"
SEED_F = EVO / "history_seed.yaml"
LEDGER_F = EVO / "ledger.jsonl"
STATE_F = EVO / "state.json"
LAST_F = EVO / "last_run.json"
LAST_MD = EVO / "last_run.md"
REPORTS = EVO / "reports"

TZ = timezone(timedelta(hours=8))


# ────────────────────────────── 基础工具 ──────────────────────────────
def now_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def canon(obj):
    """规范化后用于计算指纹：数字统一、键排序，保证同一 patch 永远同一指纹。"""
    if isinstance(obj, dict):
        return {str(k): canon(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [canon(x) for x in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def fingerprint(patch: dict) -> str:
    s = json.dumps(canon(patch), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def get_path(cfg: dict, dotted: str):
    cur = cfg
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def set_path(cfg: dict, dotted: str, value):
    ks = dotted.split(".")
    cur = cfg
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    cur[ks[-1]] = value


def apply_patch(cfg: dict, patch: dict) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in patch.items():
        set_path(out, k, v)
    return out


def read_ledger() -> list[dict]:
    if not LEDGER_F.exists():
        return []
    out = []
    for line in LEDGER_F.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[warn] ledger 有坏行，已跳过: {line[:80]}", flush=True)
    return out


def append_ledger(rec: dict):
    EVO.mkdir(exist_ok=True)
    with open(LEDGER_F, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# 只有"真裁决"才进账本。SKIP_* 是去重结果，REJECT_BUDGET / REJECT_NO_HOLDOUT 是
# 运营状态而非结论 —— 三者每周都会原样复现，写进账本只会让 12KB 的账本一年涨到 1MB，
# 还会污染 STATE.md 的裁决计数。它们保留在 reports/ 与 last_run 里可查。
NON_VERDICT = {"REJECT_BUDGET", "REJECT_NO_HOLDOUT"}


def is_verdict(decision: str) -> bool:
    return not decision.startswith("SKIP") and decision not in NON_VERDICT


def read_state() -> dict:
    if STATE_F.exists():
        return json.loads(STATE_F.read_text(encoding="utf-8"))
    return {"strategy_version": "v121.0", "policy_version": 1, "holdout_used": {}, "applied": []}


def write_state(st: dict):
    STATE_F.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_ts(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
    except Exception:
        return datetime.now(TZ) - timedelta(days=9999)


# ────────────────────────────── 回测窗口 ──────────────────────────────
def build_windows(main_dates: list[str], policy: dict) -> dict:
    hs = str(policy["windows"]["holdout_start"]).replace("-", "")
    pre = [d for d in main_dates if d < hs]
    post = [d for d in main_dates if d >= hs]
    fast_n = int(policy["windows"].get("fast_days", 1500))
    ext_n = int(policy["windows"].get("extended_days", 0) or 0)
    fast = pre[-fast_n:] if fast_n > 0 else pre
    ext = pre if ext_n <= 0 else pre[-ext_n:]
    if not fast or not ext:
        raise SystemExit(f"[fatal] 窗口为空: pre={len(pre)} post={len(post)} holdout_start={hs}")
    return {"fast": fast, "extended": ext, "holdout": post}


def yearly_returns(equity: list[dict]) -> dict[str, float]:
    """从单次回测的净值曲线按自然年分解收益 —— 不额外跑回测，是分年闸门的成本关键。"""
    pts = [(str(e["d"]), float(e["nav"])) for e in equity if e.get("nav") is not None]
    if not pts:
        return {}
    out: dict[str, float] = {}
    anchor = pts[0][1]
    cur = pts[0][0][:4]
    last = pts[0][1]
    for d, v in pts:
        y = d[:4]
        if y != cur:
            out[cur] = round((last / anchor - 1) * 100, 2) if anchor else 0.0
            anchor = last
            cur = y
        last = v
    out[cur] = round((last / anchor - 1) * 100, 2) if anchor else 0.0
    return out


def _kama(m: dict) -> float:
    v = m.get("卡玛比率")
    return float(v) if v is not None else 0.0


def _ret(m: dict) -> float:
    return float(m.get("总收益率%") or 0.0)


def _mdd(m: dict) -> float:
    return float(m.get("最大回撤%") or 0.0)


def rel_pct(cand: float, base: float) -> float:
    return (cand - base) / abs(base) * 100 if abs(base) > 1e-9 else 0.0


def run_win(cfg: dict, ctx: dict, dates: list[str], tag: str = "") -> dict:
    from src.backtest import run_backtest

    c2 = dict(ctx)
    c2["main_dates"] = dates
    t0 = time.time()
    res = run_backtest(cfg, None, c2)
    m = dict(res["metrics"])
    yr = yearly_returns(res.get("equity") or [])
    del res
    gc.collect()
    if tag:
        print(f"    · {tag:28s} {m['总收益率%']:>10.2f}% 回撤 {m['最大回撤%']:>7.2f}% "
              f"卡玛 {m['卡玛比率']} 笔 {m['交易次数']}  ({time.time()-t0:.0f}s)", flush=True)
    return {"metrics": m, "yearly": yr, "secs": round(time.time() - t0, 1)}


def diff_summary(c: dict, b: dict) -> dict:
    return {
        "ret_cand": _ret(c), "ret_base": _ret(b),
        "d_ret_pp": round(_ret(c) - _ret(b), 2),
        "d_ret_rel_pct": round(rel_pct(_ret(c), _ret(b)), 2),
        "mdd_cand": _mdd(c), "mdd_base": _mdd(b),
        "d_mdd_pp": round(_mdd(c) - _mdd(b), 2),
        "kama_cand": _kama(c), "kama_base": _kama(b),
        "d_kama": round(_kama(c) - _kama(b), 2),
        "trades_cand": int(c.get("交易次数") or 0), "trades_base": int(b.get("交易次数") or 0),
        "win_cand": c.get("胜率%"), "win_base": b.get("胜率%"),
        "avg_cand": c.get("平均单笔收益%"), "avg_base": b.get("平均单笔收益%"),
    }


# ────────────────────────────── 账本去重 ──────────────────────────────
def known_verdict(fp_: str, ledger: list[dict], cooldown_days: int):
    """返回 (记录, 判定)：命中历史拒绝则跳过；命中已采纳则跳过。"""
    cutoff = datetime.now(TZ) - timedelta(days=cooldown_days)
    for rec in reversed(ledger):
        if rec.get("fingerprint") != fp_ or rec.get("type", "decision") != "decision":
            continue
        dec = str(rec.get("decision", ""))
        if dec.startswith("REJECT"):
            # 历史回填的裁决永久有效；引擎自己的拒绝走冷却期
            if rec.get("seed") or parse_ts(rec.get("ts", "")) >= cutoff:
                return rec, "SKIP_KNOWN"
        elif dec == "PASS":
            return rec, "SKIP_APPLIED"
    return None, None


def expired(cand: dict) -> bool:
    exp = cand.get("expires")
    return bool(exp) and str(exp) < today_str()


# ────────────────────────────── run ──────────────────────────────
def cmd_run(a) -> int:
    policy = load_yaml(POLICY_F)
    raw_c = load_yaml(CANDS_F)
    cands = raw_c if isinstance(raw_c, list) else (raw_c.get("candidates") or [])
    ledger = read_ledger()
    state = read_state()
    run_id = datetime.now(TZ).strftime("%Y%m%dT%H%M%S")
    autonomy = policy.get("autonomy") or {}
    cooldown = int((policy.get("limits") or {}).get("cooldown_days", 365))
    max_cand = a.max or int((policy.get("limits") or {}).get("max_candidates_per_run", 2))

    print(f"[evolve] run_id={run_id} policy=v{policy.get('version')} 账本 {len(ledger)} 条", flush=True)

    # ── 1. 先做纯记账的筛选（不跑回测，成本为零）
    pre_records, todo = [], []
    for c in cands:
        cid, patch = c["id"], c.get("patch") or {}
        if a.only and cid != a.only:
            continue
        fp_ = fingerprint(patch)
        rec, verdict = known_verdict(fp_, ledger, cooldown)
        if verdict:
            pre_records.append({
                "candidate_id": cid, "kind": c.get("kind"), "patch": patch, "fingerprint": fp_,
                "decision": verdict, "gates": [],
                "reason": (f"账本已有裁决：{rec.get('decision')} @ {rec.get('ts')} —— {rec.get('reason','')[:80]}"
                           if verdict == "SKIP_KNOWN" else "该指纹已被采纳并应用，无需重复评估"),
            })
            print(f"  - {cid:18s} {verdict}", flush=True)
            continue
        if expired(c):
            pre_records.append({"candidate_id": cid, "kind": c.get("kind"), "patch": patch,
                                "fingerprint": fp_, "decision": "SKIP_EXPIRED", "gates": [],
                                "reason": f"候选已于 {c.get('expires')} 过期，需重新登记"})
            print(f"  - {cid:18s} SKIP_EXPIRED", flush=True)
            continue
        if autonomy.get(c.get("kind"), "review") == "never" and not a.only:
            pre_records.append({"candidate_id": cid, "kind": c.get("kind"), "patch": patch,
                                "fingerprint": fp_, "decision": "SKIP_POLICY", "gates": [],
                                "reason": f"policy.autonomy.{c.get('kind')} = never：该类变更不允许自动评估/生效"})
            print(f"  - {cid:18s} SKIP_POLICY", flush=True)
            continue
        todo.append((c, fp_))

    # 每轮评估上限：超出的不是"没通过"，而是"这周没排到"。
    # 必须显式记账（SKIP_QUOTA 只进报告、不进账本），否则候选池一长，
    # 排在后面的人会以为自己的假设被试过了。
    if len(todo) > max_cand:
        for c, fp_ in todo[max_cand:]:
            pre_records.append({"candidate_id": c["id"], "kind": c.get("kind"),
                                "patch": c.get("patch") or {}, "fingerprint": fp_,
                                "decision": "SKIP_QUOTA", "gates": [],
                                "reason": f"本轮名额 {max_cand} 已满（policy.limits.max_candidates_per_run），"
                                          f"排队到下一次运行"})
            print(f"  - {c['id']:18s} SKIP_QUOTA", flush=True)
        todo = todo[:max_cand]

    results = list(pre_records)
    no_write = a.smoke or getattr(a, "dry_run", False)
    if not todo:
        print("[evolve] 没有需要评估的候选，仅写账本", flush=True)
        if not no_write:
            for r in pre_records:
                if is_verdict(r["decision"]):
                    append_ledger({"ts": now_str(), "type": "decision", "run_id": run_id,
                                   "policy_version": policy.get("version"), **r})
        _write_last(run_id, policy, results, state, _suffix(a))
        return 0

    # ── 2. 建一次上下文，三个窗口全部从中切片
    from src.backtest import prepare_context

    hist_days = int(policy["windows"].get("history_days", 3378))
    if a.smoke:
        hist_days = 700
    print(f"[evolve] 构建上下文 {hist_days} 日（一次，供全部窗口复用）...", flush=True)
    from src.config import load_config

    cfg = load_config(str(ROOT / "config.yaml"))
    t0 = time.time()
    ctx = prepare_context(cfg, hist_days)
    md = ctx["main_dates"]
    w = build_windows(md, policy)
    print(f"[evolve] 上下文就绪 {time.time()-t0:.0f}s  {md[0]}~{md[-1]}  "
          f"fast={len(w['fast'])} ext={len(w['extended'])} holdout={len(w['holdout'])}", flush=True)

    base_cache: dict[str, dict] = {}

    def base(name: str) -> dict:
        if name not in base_cache:
            base_cache[name] = run_win(cfg, ctx, w[name], f"baseline[{name}]")
        return base_cache[name]

    # 先跑 fast 基线（快筛必需）；extended 基线等有候选通过快筛再跑
    base("fast")

    budget = policy.get("budget") or {}
    year = str(datetime.now(TZ).year)
    limit = int(budget.get("holdout_limit_per_year", 4))
    used = int((state.get("holdout_used") or {}).get(year, 0))

    for c, fp_ in todo:
        cid = c["id"]
        print(f"\n[evolve] ▶ {cid}  {c.get('patch')}", flush=True)
        gates: list[dict] = []
        evidence: dict = {}
        decision, reason = "", ""

        cand_cfg = apply_patch(cfg, c.get("patch") or {})

        # ── 第一道：快筛
        r_f = run_win(cand_cfg, ctx, w["fast"], f"{cid}[fast]")
        b_f = base("fast")
        d_f = diff_summary(r_f["metrics"], b_f["metrics"])
        d_f["secs"] = {"cand": r_f["secs"], "base": b_f["secs"]}
        evidence["fast"] = d_f
        g1 = (d_f["d_ret_rel_pct"] >= float(policy["gates"]["fast_min_rel_pct"])
              and d_f["d_kama"] >= float(policy["gates"]["fast_min_delta_kama"]))
        gates.append({"id": "G1_fast", "pass": g1,
                      "detail": f"Δ收益 {d_f['d_ret_rel_pct']:+.2f}% / Δ卡玛 {d_f['d_kama']:+.2f} / "
                                f"笔数 {d_f['trades_base']}→{d_f['trades_cand']}"})
        if not g1:
            decision, reason = "REJECT_FAST", "快筛未过：封存区前 1500 日已明确变坏，不值得进入全史口径"
            results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, False))
            continue

        # ── 第二道：主口径（封存区前全部历史）
        b_e = base("extended")
        r_e = run_win(cand_cfg, ctx, w["extended"], f"{cid}[ext]")
        d_e = diff_summary(r_e["metrics"], b_e["metrics"])
        d_e["secs"] = {"cand": r_e["secs"], "base": b_e["secs"]}
        evidence["extended"] = d_e
        g2 = (d_e["d_ret_rel_pct"] >= float(policy["gates"]["ext_min_rel_pct"])
              and d_e["d_kama"] >= float(policy["gates"]["ext_min_delta_kama"])
              and d_e["d_mdd_pp"] >= -float(policy["gates"]["ext_max_dd_worse_pp"]))
        gates.append({"id": "G2_extended", "pass": g2,
                      "detail": f"Δ收益 {d_e['d_ret_rel_pct']:+.2f}% / Δ卡玛 {d_e['d_kama']:+.2f} / "
                                f"Δ回撤 {d_e['d_mdd_pp']:+.2f}pp / 笔数 {d_e['trades_base']}→{d_e['trades_cand']}"})
        if not g2:
            # 落在噪声带里 = 这个"结构"根本没起作用（或作用被上游过滤吸收），
            # 与"起作用但更差"是两种完全不同的结论，必须分开记账。
            noise_rel = float(policy["gates"].get("noise_band_rel_pct", 3.0))
            noise_k = float(policy["gates"].get("noise_band_delta_kama", 0.15))
            if abs(d_e["d_ret_rel_pct"]) < noise_rel and abs(d_e["d_kama"]) < noise_k:
                decision = "REJECT_NOISE"
                reason = ("与基线无可测差异：该条件在现行选股/执行路径上不起作用"
                          "（可能只影响备用池，或被上游过滤吸收），无可利用信息")
            else:
                decision = "REJECT_EXTENDED"
                reason = "主口径未过：全史维度收益/卡玛/回撤至少一项不达标"
            results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, False))
            continue

        # ── 第三道：分年一致性
        yrs = sorted(set(r_e["yearly"]) & set(b_e["yearly"]))
        deltas = {y: round(r_e["yearly"][y] - b_e["yearly"][y], 2) for y in yrs}
        win_n = sum(1 for v in deltas.values() if v > 0)
        ratio = win_n / len(yrs) if yrs else 0.0
        worst_y = min(deltas, key=deltas.get) if deltas else ""
        worst_v = deltas.get(worst_y, 0.0)
        evidence["yearly_delta"] = deltas
        g3 = (ratio >= float(policy["gates"]["year_win_ratio"])
              and worst_v >= float(policy["gates"]["year_worst_delta_pp"]))
        gates.append({"id": "G3_yearly", "pass": g3,
                      "detail": f"{win_n}/{len(yrs)} 年改善({ratio:.0%})，最差 {worst_y} {worst_v:+.2f}pp"})
        if not g3:
            decision, reason = "REJECT_YEARS", "分年不一致：改善集中在少数年份，或存在单年崩塌"
            results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, False))
            continue

        # ── 第四道：邻域平台（仅当候选声明了 grid）
        grid = c.get("grid") or {}
        if grid:
            path, vals = next(iter(grid.items()))
            ok_n, det = True, []
            for v in vals:
                np_ = dict(c.get("patch") or {})
                np_[path] = v
                r_n = run_win(apply_patch(cfg, np_), ctx, w["fast"], f"{cid}[nbr {v}]")
                dk = round(_kama(r_n["metrics"]) - _kama(b_f["metrics"]), 2)
                ok_n &= dk >= float(policy["gates"]["neighbor_delta_kama"])
                det.append(f"{v}:Δ卡玛{dk:+.2f}")
            g4 = ok_n
            gates.append({"id": "G4_neighbor", "pass": g4, "detail": " / ".join(det)})
            if not g4:
                decision, reason = "REJECT_SPIKE", "邻域不成立：只有单点改善，属于孤立尖峰而非平台"
                results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, False))
                continue

        # ── 第五道：封存区（预算制）
        if not w["holdout"]:
            decision, reason = "REJECT_NO_HOLDOUT", "封存区为空（数据尚未越过 holdout_start）"
            results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, False))
            continue
        if used >= limit:
            decision, reason = "REJECT_BUDGET", f"{year} 年封存区预算已用尽（{used}/{limit}），排队到下一年"
            gates.append({"id": "G5_holdout", "pass": False, "detail": reason})
            results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, False))
            continue

        used += 1
        if not no_write:
            # 先记账再评估：即使进程崩在中途，预算也算已消耗 —— 宁可少看一次，不能多看一次
            state.setdefault("holdout_used", {})[year] = used
            write_state(state)
        b_h = base("holdout")
        r_h = run_win(cand_cfg, ctx, w["holdout"], f"{cid}[holdout]")
        d_h = diff_summary(r_h["metrics"], b_h["metrics"])
        d_h["secs"] = {"cand": r_h["secs"], "base": b_h["secs"]}
        evidence["holdout"] = d_h
        evidence["holdout_budget"] = {"year": year, "used": used, "limit": limit}
        g5 = (d_h["d_ret_rel_pct"] >= float(policy["gates"]["holdout_min_rel_pct"])
              and d_h["d_kama"] >= float(policy["gates"]["holdout_min_delta_kama"]))
        gates.append({"id": "G5_holdout", "pass": g5,
                      "detail": f"Δ收益 {d_h['d_ret_rel_pct']:+.2f}% / Δ卡玛 {d_h['d_kama']:+.2f} / "
                                f"笔数 {d_h['trades_base']}→{d_h['trades_cand']}（预算 {used}/{limit}）"})
        if not g5:
            decision, reason = "REJECT_HOLDOUT", "封存区未确认：过不了最后一道独立时段（预算已消耗）"
            results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, True))
            continue

        decision = "PASS"
        act = autonomy.get(c.get("kind"), "review")
        reason = {"auto": "全部闸门通过 → 由工作流直接应用并提交 main",
                  "review": "全部闸门通过 → 开 PR 等人工合并",
                  "never": "全部闸门通过，但该类变更 policy 禁止自动生效"}.get(act, act)
        results.append(_finish(cid, c, fp_, decision, reason, gates, evidence, True))

    if not no_write:
        n_verdict = 0
        for r in results:
            if not is_verdict(r["decision"]):
                continue
            n_verdict += 1
            append_ledger({"ts": now_str(), "type": "decision", "run_id": run_id,
                           "policy_version": policy.get("version"), **r})
        write_state(state)
        print(f"[evolve] 账本新增 {n_verdict} 条（去重/预算类记录只进报告，不进账本）", flush=True)
    _write_last(run_id, policy, results, state, _suffix(a), _win_meta(w))
    print(f"[evolve] 窗口: {_win_meta(w)}", flush=True)

    passed = [r for r in results if r["decision"] == "PASS"]
    print(f"\n[evolve] 完成：{len(passed)} 个 PASS / {len(results)} 条记录"
          f"{'（不写账本的模式）' if no_write else ''}", flush=True)
    return 0


def _suffix(a) -> str:
    return ".smoke" if getattr(a, "smoke", False) else (".dry" if getattr(a, "dry_run", False) else "")


def _finish(cid, c, fp_, decision, reason, gates, evidence, holdout_used) -> dict:
    print(f"  ⇒ {cid}: {decision} — {reason}", flush=True)
    return {"candidate_id": cid, "kind": c.get("kind"), "patch": c.get("patch") or {},
            "fingerprint": fp_, "decision": decision, "reason": reason,
            "gates": gates, "evidence": evidence, "holdout_used": holdout_used,
            "question": c.get("question", "")}


def _win_meta(w: dict) -> dict:
    return {k: {"days": len(v), "from": v[0] if v else "", "to": v[-1] if v else ""}
            for k, v in w.items()}


def _write_last(run_id: str, policy: dict, results: list[dict], state: dict,
                suffix: str = "", windows: dict | None = None):
    payload = {"run_id": run_id, "ts": now_str(), "policy_version": policy.get("version"),
               "strategy_version": state.get("strategy_version"),
               "windows": windows or {},
               "smoke": bool(suffix),
               "verdicts": results}
    last_f = EVO / f"last_run{suffix}.json"
    last_md = EVO / f"last_run{suffix}.md"
    md = render_md(payload, state)
    last_f.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    last_md.write_text(md, encoding="utf-8")
    # 归档一份不可变的报告：last_run 是"指针"（会被下次覆盖、且已 gitignore），
    # reports/ 才是历史。只有 smoke / dry 不归档，避免把假数据混进证据链。
    if not suffix:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / f"{run_id}.md").write_text(md, encoding="utf-8")
    print(f"[evolve] 报告已写: {last_md}", flush=True)


ICON = {"PASS": "✅", "REJECT_FAST": "❌", "REJECT_EXTENDED": "❌", "REJECT_YEARS": "⚠️",
        "REJECT_SPIKE": "⚠️", "REJECT_HOLDOUT": "❌", "REJECT_NOISE": "⚪",
        "REJECT_BUDGET": "🕒", "REJECT_NO_HOLDOUT": "⌛",
        "SKIP_KNOWN": "🔁", "SKIP_APPLIED": "🔁", "SKIP_EXPIRED": "⌛", "SKIP_POLICY": "🚫",
        "SKIP_QUOTA": "⏭"}


def render_md(payload: dict, state: dict) -> str:
    w = payload.get("windows") or {}
    wline = " · ".join(f"{k} {v['days']}日({v['from']}~{v['to']})" for k, v in w.items())
    lines = [f"## 自进化裁决 · {payload['ts']}",
             "",
             f"`run_id` `{payload['run_id']}` · policy v{payload['policy_version']} · "
             f"策略版本 `{payload.get('strategy_version')}`",
             ""]
    if wline:
        lines += [f"窗口：{wline}", ""]
    lines += ["| 候选 | 类型 | 裁决 | 快筛 Δ收益 | 主口径 Δ收益 | 主口径 Δ卡玛 | 封存区 Δ收益 |",
              "|---|---|---|---|---|---|---|"]
    for r in payload["verdicts"]:
        ev = r.get("evidence") or {}
        def g(k, f):
            return ev.get(k, {}).get(f)
        def fmt(v, suffix="%"):
            return "—" if v is None else f"{v:+.2f}{suffix}"
        lines.append(f"| `{r['candidate_id']}` | {r.get('kind','')} | {ICON.get(r['decision'],'')} {r['decision']} | "
                     f"{fmt(g('fast','d_ret_rel_pct'))} | {fmt(g('extended','d_ret_rel_pct'))} | "
                     f"{fmt(g('extended','d_kama'))} | {fmt(g('holdout','d_ret_rel_pct'))} |")
    lines.append("")
    # 绝对基线：只有 Δ 没有 base，读者无法判断"−62%"是从多高掉下来的。
    # 全 SKIP 的空跑没有证据，这一段整体不出现（否则表格下面挂两个空行）。
    base_lines = []
    for key, label in (("fast", "快筛"), ("extended", "主口径"), ("holdout", "封存区")):
        for r in payload["verdicts"]:
            e = (r.get("evidence") or {}).get(key)
            if e:
                base_lines.append(f"基线（{label}窗口）：{e['ret_base']:+.2f}% / 回撤 {e['mdd_base']:.2f}% / "
                                  f"卡玛 {e['kama_base']:.2f} / {e['trades_base']} 笔")
                break
    if base_lines:
        lines += base_lines   # 表格尾部已有一个空行，这里不再补
    lines.append("")
    for r in payload["verdicts"]:
        lines.append(f"**`{r['candidate_id']}`** — {r['reason']}")
        if r.get("question"):
            lines.append(f"> {r['question']}")
        for g_ in r.get("gates") or []:
            lines.append(f"- {'✅' if g_['pass'] else '❌'} `{g_['id']}` {g_['detail']}")
        lines.append("")
    used = (state.get("holdout_used") or {})
    lines.append(f"封存区预算：{used}（上限见 `evolution/policy.yaml`）")
    lines += ["",
              "> 口径提醒：快筛窗口止于封存区前一日，与 README 首页的「最近 1500 日」"
              "（含封存区）不是同一段行情，两处数字不可互引；能横向比较的只有本表内的 Δ。"]
    return "\n".join(lines) + "\n"


# ────────────────────────────── apply ──────────────────────────────
_FINAL_KEY = re.compile(r"^(?P<ind>[ \t]*)(?P<key>[A-Za-z0-9_]+):(?P<sp>[ \t]*)"
                        r"(?P<val>[^#\n]*?)(?P<tail>[ \t]*(?:#.*)?)$")
_TOP_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:")


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if v is None:
        return "null"
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        # 列表/字典一律写成 flow 风格单行：config.yaml 里这些块本来就没有逐项注释，
        # 单行化不会丢信息，且能避免"删子块时误删下一节的注释"。
        dumped = yaml.safe_dump(v, default_flow_style=True, allow_unicode=True, sort_keys=False)
        return dumped.strip().split("\n")[0]
    raise TypeError(f"不支持的类型 {type(v)}: {v}")


def _block(lines: list[str], lo: int, hi: int, key: str, indent: int):
    """在 [lo,hi) 内找出缩进为 indent 的 `key:` 块，返回 (行号, 子块结束行号)。"""
    pat = re.compile(rf"^[ \t]*{re.escape(key)}:[ \t]*(?:#.*)?$")
    for i in range(lo, hi):
        m = pat.match(lines[i])
        if not m:
            continue
        ind = len(lines[i]) - len(lines[i].lstrip())
        if ind != indent:
            continue
        j = i + 1
        while j < hi:
            s = lines[j]
            if s.strip() and not s.lstrip().startswith("#"):
                if len(s) - len(s.lstrip()) <= indent:
                    break
            j += 1
        return i, j
    return None


def patch_config_text(text: str, dotted: str, value) -> str:
    """按缩进定位并替换单个标量，保留全部注释与其余排版（config.yaml 有 95KB 注释，不能 yaml.dump）。"""
    lines = text.split("\n")
    parts = dotted.split(".")
    lo, hi, indent = 0, len(lines), 0
    for k in parts[:-1]:
        found = _block(lines, lo, hi, k, indent)
        if not found:
            raise KeyError(f"config.yaml 中找不到节 `{k}`（路径 {dotted}）")
        lo, hi = found[0] + 1, found[1]
        indent = None
        for i in range(lo, hi):
            s = lines[i]
            if s.strip() and not s.lstrip().startswith("#"):
                indent = len(s) - len(s.lstrip())
                break
        if indent is None:
            raise KeyError(f"节 `{k}` 为空，无法写入 {dotted}")

    key = parts[-1]
    hits = []
    for i in range(lo, hi):
        m = _FINAL_KEY.match(lines[i])
        if m and m.group("key") == key and len(m.group("ind")) == indent:
            hits.append((i, m))
    if not hits:
        raise KeyError(f"config.yaml 中找不到键 `{dotted}`")
    if len(hits) > 1:
        raise KeyError(f"`{dotted}` 在文件中出现 {len(hits)} 次，拒绝猜测")
    i, m = hits[0]
    # 列表/字典要写成单行 flow 风格时，先清掉原来的 "- item" 子块（只删列表项，不碰注释）
    if isinstance(value, (list, dict)):
        j = i + 1
        while j < hi:
            s = lines[j]
            if s.strip() == "":
                k = j
                while k < hi and lines[k].strip() == "":
                    k += 1
                if (k < hi and re.match(r"^[ \t]*-[ \t]", lines[k])
                        and len(lines[k]) - len(lines[k].lstrip()) > indent):
                    j = k
                    continue
                break
            if (re.match(r"^[ \t]*-[ \t]", s)
                    and len(s) - len(s.lstrip()) > indent):
                j += 1
                continue
            break
        del lines[i + 1:j]
    sp = m.group("sp") or " "   # `key:` 后必须留空格，否则 flow 风格的 `key:[]` 不是合法 YAML
    lines[i] = f"{m.group('ind')}{key}:{sp}{_fmt_scalar(value)}{m.group('tail')}"
    out = "\n".join(lines)
    return out if out.endswith("\n") else out + "\n"


def cmd_apply(a) -> int:
    from src.config import load_config

    ledger = read_ledger()
    state = read_state()
    target = None
    for rec in reversed(ledger):
        if rec.get("candidate_id") == a.id and rec.get("decision") == "PASS":
            target = rec
            break
    if not target:
        print(f"[apply] 账本里没有 `{a.id}` 的 PASS 记录，拒绝应用", flush=True)
        return 2
    applied = {x.get("fingerprint") for x in ledger if x.get("type") == "apply"}
    if target["fingerprint"] in applied and not a.force:
        print(f"[apply] `{a.id}` 已应用过（fingerprint {target['fingerprint']}），跳过", flush=True)
        return 0

    cfg_path = ROOT / "config.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    for k, v in target["patch"].items():
        text = patch_config_text(text, k, v)

    # 写入前置自检：先写临时文件并解析校验，避免把 config 改坏
    tmp = cfg_path.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        new_cfg = load_config(str(tmp))
        for k, v in target["patch"].items():
            got = get_path(new_cfg, k)
            assert got == v or (isinstance(v, (int, float)) and float(got) == float(v)), \
                f"{k}: 期望 {v} 实际 {got}"
        assert "selection" in new_cfg and "execution" in new_cfg
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"[apply] 校验失败，config 未改动: {e}", flush=True)
        return 3

    ver = str(state.get("strategy_version", "v121.0"))
    try:
        base, minor = ver.lstrip("v").split(".")
        new_ver = f"v{base}.{int(minor) + 1}"
    except Exception:
        new_ver = ver + ".1"
    header = (f"# [EVOLVE] {new_ver} | {today_str()} | {a.id} | "
              f"ledger-fp={target['fingerprint']} | 由 tools/evolve.py 应用\n")
    text = header + text
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(cfg_path)

    state["strategy_version"] = new_ver
    state.setdefault("applied", []).append({"ts": now_str(), "candidate_id": a.id,
                                            "fingerprint": target["fingerprint"],
                                            "patch": target["patch"], "version": new_ver})
    write_state(state)
    append_ledger({"ts": now_str(), "type": "apply", "candidate_id": a.id,
                   "fingerprint": target["fingerprint"], "patch": target["patch"],
                   "version": new_ver})
    print(f"[apply] {a.id} 已应用到 config.yaml，策略版本 {ver} → {new_ver}", flush=True)
    print(f"[apply] patch: {json.dumps(target['patch'], ensure_ascii=False)}", flush=True)
    return 0


# ────────────────────────────── seed / state ──────────────────────────────
def cmd_seed(_a) -> int:
    seed = load_yaml(SEED_F).get("entries") or []
    ledger = read_ledger()
    have = {(r.get("fingerprint"), r.get("decision")) for r in ledger}
    n = 0
    for e in seed:
        fp_ = fingerprint(e["patch"])
        if (fp_, e["decision"]) in have:
            continue
        append_ledger({"ts": f"{today_str()} 00:00:00", "type": "decision", "seed": True,
                       "run_id": "seed", "policy_version": 1, "candidate_id": e["id"],
                       "kind": e.get("kind"), "patch": e["patch"], "fingerprint": fp_,
                       "decision": e["decision"], "reason": e.get("reason", ""),
                       "round": e.get("round"), "gates": [], "evidence": {}})
        have.add((fp_, e["decision"]))
        n += 1
    print(f"[seed] 新增 {n} 条历史裁决，账本共 {len(read_ledger())} 条", flush=True)
    return 0


def cmd_passes(a) -> int:
    """打印最近一次裁决里 PASS 的候选 id（供工作流决定要不要开 PR）。"""
    if not LAST_F.exists():
        return 0
    try:
        payload = json.loads(LAST_F.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    ids = []
    for v in payload.get("verdicts", []):
        if v.get("decision") != "PASS":
            continue
        kind = v.get("kind")
        if a.kind and kind != a.kind:
            continue
        if a.exclude_kind and kind == a.exclude_kind:
            continue
        ids.append(v["candidate_id"])
    print(" ".join(ids))
    return 0


def cmd_state(_a) -> int:
    policy = load_yaml(POLICY_F)
    state = read_state()
    ledger = read_ledger()
    decisions = [r for r in ledger if r.get("type", "decision") == "decision"]
    lines = ["# 自进化状态", "",
             f"- 策略版本：`{state.get('strategy_version')}`",
             f"- 治理策略：v{policy.get('version')}（改它需人工 PR）",
             f"- 封存区起点：`{policy['windows']['holdout_start']}`",
             f"- 封存区预算：{state.get('holdout_used')}（上限 {policy['budget']['holdout_limit_per_year']}/年）",
             f"- 账本条目：{len(decisions)} 条（含历史回填 {sum(1 for r in decisions if r.get('seed'))} 条）",
             "",
             "## 最近 20 条裁决", "",
             "| 时间 | 候选 | 类型 | 裁决 | 理由 |", "|---|---|---|---|---|"]
    for r in decisions[-20:][::-1]:
        lines.append(f"| {str(r.get('ts',''))[:10]} | `{r.get('candidate_id')}` | {r.get('kind','')} | "
                     f"{r.get('decision')} | {str(r.get('reason',''))[:60]} |")
    (EVO / "STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="策略自进化引擎")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="裁决候选池")
    p_run.add_argument("--max", type=int, default=0, help="本次最多评估几个（0=按 policy）")
    p_run.add_argument("--only", default="", help="只评估指定 id（会绕过 autonomy=never 的跳过）")
    p_run.add_argument("--smoke", action="store_true", help="缩小历史窗口，仅验证管道连通")
    p_run.add_argument("--dry-run", action="store_true", help="完整评估但不写账本、不消耗封存区预算")
    p_run.set_defaults(func=cmd_run)

    p_ap = sub.add_parser("apply", help="把 PASS 记录写入 config.yaml")
    p_ap.add_argument("--id", required=True)
    p_ap.add_argument("--force", action="store_true")
    p_ap.set_defaults(func=cmd_apply)

    sub.add_parser("seed", help="回填历史裁决").set_defaults(func=cmd_seed)
    sub.add_parser("state", help="重写 STATE.md").set_defaults(func=cmd_state)

    p_pass = sub.add_parser("passes", help="打印最近一次裁决里 PASS 的候选 id")
    p_pass.add_argument("--kind", default="")
    p_pass.add_argument("--exclude-kind", default="")
    p_pass.set_defaults(func=cmd_passes)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
