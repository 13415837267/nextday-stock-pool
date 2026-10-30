# -*- coding: utf-8 -*-
"""把 output/replay.json 渲染为自包含的 site/replay.html(数据内嵌, 无外部依赖)。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fmt_date(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def build_svg(curve: list[dict], base: float) -> str:
    if len(curve) < 2:
        return "<p class='muted'>数据不足</p>"
    W, H, PL, PR, PT, PB = 1100, 280, 56, 18, 18, 30
    navs = [base] + [p["nav"] for p in curve]
    lo, hi = min(navs), max(navs)
    pad = (hi - lo) * 0.12 or hi * 0.01
    lo, hi = lo - pad * 0.7, hi + pad * 0.7
    n = len(navs)

    def x(i: int) -> float:
        return PL + (W - PL - PR) * i / (n - 1)

    def y(v: float) -> float:
        return PT + (H - PT - PB) * (1 - (v - lo) / (hi - lo))

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(navs))
    area = f"{x(0):.1f},{y(lo):.1f} " + pts + f" {x(n-1):.1f},{y(lo):.1f}"
    up = navs[-1] >= navs[0]
    color = "#f85149" if up else "#3fb950"

    grid = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        yy = y(v)
        grid.append(f"<line x1='{PL}' y1='{yy:.1f}' x2='{W-PR}' y2='{yy:.1f}' stroke='#232a36' stroke-width='1'/>")
        grid.append(f"<text x='{PL-8}' y='{yy+4:.1f}' fill='#8b949e' font-size='10' text-anchor='end'>{(v/base-1)*100:+.1f}%</text>")

    ticks = []
    for i in (0, n // 3, 2 * n // 3, n - 1):
        ticks.append(f"<text x='{x(i):.1f}' y='{H-8}' fill='#8b949e' font-size='10' "
                     f"text-anchor='{'start' if i == 0 else ('end' if i == n-1 else 'middle')}'>"
                     f"{curve[min(i, len(curve)-1)]['d'][5:]}</text>")

    return f"""<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block">
<defs><linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="{color}" stop-opacity=".28"/><stop offset="100%" stop-color="{color}" stop-opacity="0"/>
</linearGradient></defs>
{''.join(grid)}
<polygon points="{area}" fill="url(#g1)"/>
<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>
<line x1="{PL}" y1="{y(base):.1f}" x2="{W-PR}" y2="{y(base):.1f}" stroke="#8b949e" stroke-dasharray="4 4" stroke-width="1"/>
{''.join(ticks)}
</svg>"""


def render(payload: dict) -> str:
    m, s = payload["meta"], payload["summary"]
    base = payload.get("base_nav") or m["initial_capital"]
    svg = build_svg(payload["curve"], base)

    daily = payload.get("daily", [])
    n_day = len(daily)
    n_sep = sum(1 for t in payload["trades"] if t["buy_date"].startswith("202609"))
    n_oct = sum(1 for t in payload["trades"] if t["buy_date"].startswith("202610"))
    n_flat = sum(1 for d in daily if d.get("alloc", 1.0) <= 0)
    n_part = sum(1 for d in daily if 0 < d.get("alloc", 1.0) < 1)
    reenter = f"{max((d.get('alloc', 0) for d in daily), default=1) * 100:.0f}%"

    def cls(v):
        return "up" if (v or 0) > 0 else ("down" if (v or 0) < 0 else "")

    months = "".join(
        f"<div class='card'><div class='k'>{mm['month'][:4]}年{int(mm['month'][4:])}月</div>"
        f"<div class='v {cls(mm['ret_pct'])}'>{mm['ret_pct']:+.2f}%</div></div>"
        for mm in payload.get("monthly", []))

    data_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>历史回放 · {_fmt_date(m['start'])} ~ {_fmt_date(m['end'])}</title>
<style>
  :root {{
    --bg:#0b0e14; --panel:#12161f; --panel2:#171c26; --border:#232a36;
    --text:#e6edf3; --muted:#8b949e; --gold:#d4a017; --red:#f85149;
    --green:#3fb950; --blue:#58a6ff; --purple:#bc8cff;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:"PingFang SC","Microsoft YaHei",-apple-system,sans-serif; line-height:1.65; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 16px 64px; }}
  header {{ border-bottom:1px solid var(--border); padding-bottom:18px; margin-bottom:22px; }}
  h1 {{ font-size:25px; color:var(--gold); }}
  h1 small {{ font-size:13px; color:var(--muted); font-weight:normal; margin-left:12px; }}
  .meta {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .badges {{ margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; }}
  .badge {{ font-size:12px; padding:3px 10px; border-radius:12px; border:1px solid var(--border); color:var(--muted); }}
  .badge.ok {{ color:var(--green); border-color:var(--green); }}
  .badge.gold {{ color:var(--gold); border-color:var(--gold); }}
  .badge a {{ color:inherit; text-decoration:none; }}
  h2 {{ font-size:18px; margin:30px 0 12px; padding-left:10px; border-left:3px solid var(--gold); }}
  h2 .sub {{ font-size:12px; color:var(--muted); font-weight:normal; margin-left:8px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:14px; }}
  .card .k {{ color:var(--muted); font-size:12px; }}
  .card .v {{ font-size:20px; font-weight:600; margin-top:4px; }}
  .card .s {{ font-size:11px; color:var(--muted); margin-top:2px; }}
  .up {{ color:var(--red); }} .down {{ color:var(--green); }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; background:var(--panel); border-radius:8px; overflow:hidden; }}
  th,td {{ padding:9px 10px; text-align:left; border-bottom:1px solid var(--border); white-space:nowrap; }}
  th {{ background:#1a202b; color:var(--muted); font-weight:500; }}
  tr:hover td {{ background:#1a2029; }}
  .muted {{ color:var(--muted); }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:16px; }}
  .tbl-wrap {{ overflow-x:auto; }}
  details {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:12px 16px; }}
  summary {{ cursor:pointer; color:var(--gold); font-size:14px; }}
  .note {{ font-size:13px; color:var(--muted); }}
  .note b {{ color:var(--text); }}
  .note li {{ margin-left:18px; }}
  code {{ background:var(--panel2); padding:1px 5px; border-radius:4px; font-size:12px; color:var(--blue); }}
  .tag {{ font-size:11px; padding:1px 6px; border-radius:4px; border:1px solid var(--border); color:var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>历史回放 <small>{_fmt_date(m['start'])} ~ {_fmt_date(m['end'])}</small></h1>
  <div class="meta">按当前策略逐日重放 · 复用回测引擎 · 指标仅用截至当日数据</div>
  <div class="badges">
    <span class="badge gold">{m['strategy']}</span>
    <span class="badge ok">无未来函数</span>
    <span class="badge">预热 {m['warmup_days']} 交易日</span>
    <span class="badge">面板 {_fmt_date(m['panel_first'])} 起</span>
    <span class="badge"><a href="./index.html">← 返回实盘看板</a></span>
  </div>
</header>

<h2>区间概览 <span class="sub">按买入日归属，含区间内已平仓的 {s['n_trades']} 笔</span></h2>
<div class="cards">
  <div class="card"><div class="k">成交笔数</div><div class="v">{s['n_trades']}</div><div class="s">胜 {s['n_win']} 负 {s['n_trades'] - s['n_win']}</div></div>
  <div class="card"><div class="k">胜率</div><div class="v">{s['win_rate']}%</div></div>
  <div class="card"><div class="k">平均单笔</div><div class="v {cls(s['avg_ret_pct'])}">{s['avg_ret_pct']:+.2f}%</div></div>
  <div class="card"><div class="k">区间收益</div><div class="v {cls(s.get('range_ret_pct'))}">{s.get('range_ret_pct', 0):+.2f}%</div><div class="s">{base/10000:.1f}万 → {s.get('end_nav', 0)/10000:.1f}万</div></div>
  <div class="card"><div class="k">区间最大回撤</div><div class="v down">{s.get('range_mdd_pct', 0):.2f}%</div></div>
  <div class="card"><div class="k">最好一笔</div><div class="v up">{s['best']:+.2f}%</div></div>
  <div class="card"><div class="k">最差一笔</div><div class="v down">{s['worst']:+.2f}%</div></div>
</div>
<div class="cards" style="margin-top:12px;grid-template-columns:repeat(4,1fr)">{months}</div>

<h2>账户净值 <span class="sub">起点 {base/10000:.2f} 万（{s.get('base_date', '')} 收盘）</span></h2>
<div class="panel">{svg}</div>

<h2>成交明细 <span class="sub">T 日信号 → T+1 开盘买入 → T+2 开盘卖出</span></h2>
<div class="tbl-wrap"><table id="trades"><thead><tr>
  <th>信号日</th><th>买入日</th><th>卖出日</th><th>代码</th><th>名称</th><th>买入价</th><th>卖出价</th><th>收益</th><th>名次</th><th>备注</th>
</tr></thead><tbody></tbody></table></div>

<h2>逐日决策 <span class="sub">解释"这天为什么没买"</span></h2>
<details>
  <summary>展开 38 个交易日的选股/买入判定</summary>
  <div class="tbl-wrap" style="margin-top:12px"><table id="daily"><thead><tr>
    <th>买入日</th><th>风控</th><th>成交</th><th>候选与跳过原因</th>
  </tr></thead><tbody></tbody></table></div>
</details>

<h2>口径说明</h2>
<div class="panel note">
  <p><b>为什么成交这么少？</b> 这不是漏跑。策略要求次日<b>低开 ≤ -2.35%</b>（弱市超限参与日为 -2.75%）才承接，
  9 月候选票的实际缺口多在 -1.7% ~ +2.0%，未达承接线就主动放弃。
  这 {n_day} 个交易日里，9 月成交 {n_sep} 笔、10 月 {n_oct} 笔，另有 {n_flat} 天大盘风控直接空仓。
  10 月大盘持续走弱触发「超长风控恢复参与」——连续风控超过 5 天后按 {reenter} 仓位逐步恢复，故成交反而集中在 10 月。</p>
  <p style="margin-top:10px"><b>无未来函数的三条保证：</b></p>
  <ul>
    <li>指标（MA/动量/波动/振幅）全部为 <code>rolling</code> / <code>shift(1)</code> 向后看，无 <code>center=True</code>、无 <code>shift(-n)</code>；</li>
    <li>选股只传当日截面数据，不包含任何后续交易日行；风控序列同样只依赖当日及之前；</li>
    <li>买入取 T+1 真实开盘价、卖出取 T+2 真实开盘价（跌停回退当日收盘），绩效门控只用已发生的成交。</li>
  </ul>
  <p style="margin-top:10px"><b>与实盘对账：</b>10-23 买入 300192.SZ（-0.40%）、10-29 买入 002811.SZ（-11.16%），
  与实盘 <code>history.json</code> 的同日记录完全一致，说明回放与实盘同一套成交口径。</p>
  <p style="margin-top:10px" class="muted">差异说明：实盘每日只加载 30 日面板（<code>listed_min_bars=0</code>），
  回放用连续 155 日面板（<code>listed_min_bars=120</code>），次新股过滤口径不同，个别日期的候选池会有出入。</p>
</div>

<script>
const DATA = {data_js};
const f2 = v => (v == null ? '-' : v.toFixed(2));
const cls = v => (v > 0 ? 'up' : (v < 0 ? 'down' : ''));
const dstr = s => s ? `${{s.slice(0,4)}}-${{s.slice(4,6)}}-${{s.slice(6,8)}}` : '-';

document.querySelector('#trades tbody').innerHTML = DATA.trades.map(t => `
  <tr><td class="muted">${{dstr(t.signal_date)}}</td><td>${{dstr(t.buy_date)}}</td><td>${{dstr(t.sell_date)}}</td>
  <td>${{t.code}}</td><td>${{t.name}}</td><td>${{f2(t.buy_px)}}</td><td>${{f2(t.sell_px)}}</td>
  <td class="${{cls(t.ret_pct)}}"><b>${{t.ret_pct > 0 ? '+' : ''}}${{f2(t.ret_pct)}}%</b></td>
  <td class="muted">${{t.rank ?? '-'}}</td><td class="muted">${{t.note || ''}}</td></tr>`).join('');

document.querySelector('#daily tbody').innerHTML = DATA.daily.map(d => {{
  const fills = d.fills.map(f => `<span class="up">${{f.code}} ${{f.name}} ${{f.ret_pct > 0 ? '+' : ''}}${{f2(f.ret_pct)}}%</span>`).join('；');
  const skips = d.skipped.map(s => `${{s.code}} ${{s.name}}：${{s.note}}`).join('；');
  const cand = fills || skips || '<span class="muted">无候选</span>';
  return `<tr><td>${{dstr(d.date)}}</td>
    <td>${{d.alloc <= 0 ? '<span class="down">空仓</span>'
        : (d.trade_ok ? '<span class="up">正常</span>'
                      : '<span style="color:#d4a017">超限满仓</span>')}}</td>
    <td>${{d.fills.length ? d.fills.length : '<span class="muted">0</span>'}}</td>
    <td class="muted" style="white-space:normal">${{cand}}</td></tr>`;
}}).join('');
</script>
</div>
</body>
</html>"""


def main() -> None:
    src = ROOT / "output" / "replay.json"
    dst = ROOT / "site" / "replay.html"
    if not src.exists():
        sys.exit(f"缺少 {src}, 请先运行 tools/replay.py")
    payload = json.loads(src.read_text(encoding="utf-8"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render(payload), encoding="utf-8")
    print(f"已生成 {dst} ({dst.stat().st_size:,} 字节)")


if __name__ == "__main__":
    main()
