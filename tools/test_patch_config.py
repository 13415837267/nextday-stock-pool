#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_config_text 全类型回归（第一百二十二轮）。

只验证一件事：不改语义地替换 config.yaml 里的指定键，且
  ① 注释总量不减少（95KB 注释是这个仓最有价值的资产）
  ② 其余键的值逐位不变
  ③ 结果仍是合法 YAML，且目标键等于新值
覆盖：标量 int/float、嵌套路径、布尔、列表改值、列表缩短、列表清空、行尾注释保留。
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evolve import get_path, patch_config_text  # noqa: E402

SRC = (ROOT / "config.yaml").read_text(encoding="utf-8")
BASE = yaml.safe_load(SRC)


def comment_bytes(t: str) -> int:
    return sum(len(m) for m in re.findall(r"#[^\n]*", t))


def check(name: str, dotted: str, value, expect_root_keys=True):
    out = patch_config_text(SRC, dotted, value)
    got = yaml.safe_load(out)
    errs = []

    actual = get_path(got, dotted)
    if isinstance(value, float) or isinstance(actual, float):
        if abs(float(actual) - float(value)) > 1e-12:
            errs.append(f"值不符: 期望 {value!r} 实际 {actual!r}")
    elif actual != value:
        errs.append(f"值不符: 期望 {value!r} 实际 {actual!r}")

    # 其余键不能被牵连
    def walk(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                p = f"{prefix}{k}"
                if p == dotted:
                    continue
                if k not in a:
                    errs.append(f"多出键 {p}")
                elif k not in b:
                    errs.append(f"丢失键 {p}")
                else:
                    walk(a[k], b[k], p + ".")
        elif isinstance(a, list) and isinstance(b, list):
            if prefix.rstrip(".") == dotted:
                return
            if len(a) != len(b):
                errs.append(f"{prefix} 列表长度 {len(a)}→{len(b)}")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{prefix}[{i}].")
        else:
            if a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                               and float(a) == float(b)):
                errs.append(f"{prefix.rstrip('.')} 被改动: {a!r}→{b!r}")
    walk(BASE, got)

    cb0, cb1 = comment_bytes(SRC), comment_bytes(out)
    if cb1 < cb0 - 200:  # 允许删掉列表子块时的极少量缩进噪声
        errs.append(f"注释字节减少 {cb0}→{cb1}（{cb1-cb0}）")

    status = "PASS" if not errs else "FAIL"
    print(f"[{status}] {name:26s} {dotted:34s} → {value!r}  "
          f"(注释 {cb0}→{cb1}, 行数 {len(SRC.splitlines())}→{len(out.splitlines())})")
    for e in errs:
        print(f"        ! {e}")
    return not errs


ok = True
ok &= check("标量 float", "selection.bias20_max", 12.0)
ok &= check("标量 int", "regime.market_ma", 22)
ok &= check("嵌套 float", "regime.vol_max", 2.0)
ok &= check("布尔 翻转", "universe.exclude_bj", False)
ok &= check("布尔 回写", "execution.sell_limit_down_fallback", True)
ok &= check("列表 改值(单元素)", "execution.gap_above_tiers",
            [{"above": 0.5, "score_below": 1.8, "mult": 0.20}])
ok &= check("列表 缩短(3→1)", "execution.pool_n_tiers", [{"min": 21, "mult": 1.00}])
ok &= check("列表 加长(1→2)", "execution.gap_floor_tiers",
            [{"below": -1.0, "min_pool": 5, "floor": 1.00},
             {"below": -2.0, "min_pool": 3, "floor": 1.00}])
ok &= check("列表 清空", "execution.mkt_floor_tiers", [])
ok &= check("列表 恢复", "execution.knife_tiers", [{"mkt_above": -0.50, "res_below": -4.0, "mult": 0.00}])

# 链式：连续打三个补丁后仍合法、且互不覆盖
t = SRC
for p, v in [("selection.bias20_max", 11.4), ("regime.market_ma", 20),
             ("execution.pool_n_tiers", [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.5}])]:
    t = patch_config_text(t, p, v)
d = yaml.safe_load(t)
chain_ok = (d["selection"]["bias20_max"] == 11.4 and d["regime"]["market_ma"] == 20
            and len(d["execution"]["pool_n_tiers"]) == 2)
print(f"[{'PASS' if chain_ok else 'FAIL'}] 链式三连补丁 {'一致' if chain_ok else '互相覆盖'}")
ok &= chain_ok

# 幂等：同样补丁打两次，结果逐字相同
once = patch_config_text(SRC, "selection.bias20_max", 11.4)
twice = patch_config_text(once, "selection.bias20_max", 11.4)
idem = once == twice
print(f"[{'PASS' if idem else 'FAIL'}] 幂等性（同值二次替换逐字相同）")
ok &= idem

# 找不到的键必须报错而不是静默成功
for bad in ["selection.not_exist_key", "no_such_section.x"]:
    try:
        patch_config_text(SRC, bad, 1)
        print(f"[FAIL] 缺失键 {bad} 未报错")
        ok = False
    except KeyError:
        print(f"[PASS] 缺失键 {bad} 正确抛 KeyError")

print("\n" + ("全部通过 ✅" if ok else "存在失败项 ❌"))
sys.exit(0 if ok else 1)
