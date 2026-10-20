# -*- coding: utf-8 -*-
"""配置加载与板块工具。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def board_of(ts_code: str) -> str | None:
    """按代码前缀判断板块。"""
    code = ts_code.split(".")[0]
    suffix = ts_code.split(".")[-1] if "." in ts_code else ""
    if suffix == "BJ":
        return "北交所"
    if suffix == "SH":
        return "科创板" if code.startswith("688") else "主板"
    if suffix == "SZ":
        return "创业板" if code.startswith("30") else "主板"
    return None
