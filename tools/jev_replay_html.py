"""jev_replay_html.py — render an OpenJev match to one self-contained HTML replay.

Reads a columnar replay (`.npz`) plus the match's OpenJev decision log
(`<tag>.openjev.jsonl`, found next to the replay by default) and writes an
HTML page with the pitch animation on the left and, synced to the playhead,
OpenJev's current decision on the right: chosen split, confidence, the full
probability bar chart and which tactic each of our robots is running. No
external scripts or fonts — open it in any browser, or send it to someone.

Usage:
    pixi run python -m tools.jev_replay_html replays/jev_smoke/openjev_vs_tiki_taka_Rk.npz
    pixi run python -m tools.jev_replay_html <npz> --log <openjev.jsonl> --out match.html --stride 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_reader import load_columnar_replay
from utama_core.strategy.openjev.splits import SPLIT_KEYS, TACTIC_GLOSSARY


def _r(a: np.ndarray, nd: int = 3) -> list:
    return np.round(np.nan_to_num(a, nan=0.0), nd).tolist()


def build_payload(npz: Path, log: Path | None, stride: int) -> dict:
    rep = load_columnar_replay(npz)
    idx = np.arange(0, rep.n_ticks, stride)
    cmd_names = {c.value: c.name for c in RefereeCommand}

    scores = []
    for tick in sorted(rep.sparse_referee):
        ref = rep.sparse_referee[tick]
        scores.append([float(rep.ts[tick]), ref.yellow_team.score, ref.blue_team.score])

    decisions = []
    if log is not None and log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            decisions.append(
                {
                    "t": row["t"],
                    "choice": row.get("choice"),
                    "confidence": row.get("confidence"),
                    "probs": row.get("probabilities") or {},
                    "partition": row.get("partition") or {},
                    "latency_ms": row.get("latency_ms"),
                    "cached": row.get("cached"),
                    "fallback": row.get("fallback"),
                    "possession": (row.get("live") or {})
                    .get("possession", {})
                    .get("side"),
                }
            )

    result = None
    res_path = npz.with_name(npz.name.replace(".npz", ".result.json"))
    if res_path.exists():
        result = json.loads(res_path.read_text())

    dims = STANDARD_FIELD_DIMS
    return {
        "title": npz.stem,
        "field": {
            "hl": dims.full_field_half_length,
            "hw": dims.full_field_half_width,
            "dd": 2 * dims.half_defense_area_depth,
            "dw": dims.half_defense_area_width,
            "gw": dims.half_goal_width,
            "cc": dims.center_circle_radius,
        },
        "openjev_is_yellow": bool(rep.my_team_is_yellow),
        "openjev_is_right": bool(rep.my_team_is_right),
        "friendly_ids": rep.friendly_ids.tolist(),
        "enemy_ids": rep.enemy_ids.tolist(),
        "ts": _r(rep.ts[idx]),
        "fp": _r(rep.friendly_p[idx]),
        "fo": _r(rep.friendly_orientation[idx], 2),
        "ep": _r(rep.enemy_p[idx]),
        "eo": _r(rep.enemy_orientation[idx], 2),
        "bp": _r(rep.ball_p[idx][:, :2]),
        "cmd": [
            cmd_names.get(int(c), "") if c >= 0 else ""
            for c in rep.referee_command[idx]
        ],
        "scores": scores,
        "decisions": decisions,
        "split_keys": SPLIT_KEYS,
        "tactics": list(TACTIC_GLOSSARY),
        "result": result,
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenJev Match Replay</title>
<style>
:root{--bg:#f6f5f1;--panel:#ffffff;--ink:#1d1d1b;--muted:#6b6a64;--line:#dcdad2;--pitch:#2f7d4f;--pitchline:#e9f3ec;
--yellow:#e3b505;--blue:#2d6cdf;--ball:#f06b1f;--bar:#2d6cdf;--barbg:#ecebe5;--pick:#e3b505;
--t-attack:#d9483b;--t-overload:#b04fc4;--t-press:#e98b1a;--t-mark:#2d6cdf;--t-block:#1f9e8f;--t-clear:#7a7a72;--t-goalkeeper:#444;--t-none:#999}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--panel:#1f1f1d;--ink:#ecebe5;--muted:#a3a198;
--line:#34332f;--pitch:#245f3c;--pitchline:#cfe3d6;--barbg:#2a2a27}}
:root[data-theme="dark"]{--bg:#141413;--panel:#1f1f1d;--ink:#ecebe5;--muted:#a3a198;--line:#34332f;--pitch:#245f3c;--pitchline:#cfe3d6;--barbg:#2a2a27}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 16px 6px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline}
h1{font-size:17px;margin:0}.muted{color:var(--muted)}.score{font-variant-numeric:tabular-nums;font-weight:600;font-size:16px}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(280px,1fr);gap:14px;padding:8px 16px 16px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
canvas{width:100%;height:auto;display:block;border-radius:6px}
.controls{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
button,select{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
input[type=range]{flex:1;min-width:140px}
.row{display:grid;grid-template-columns:118px 1fr 44px;gap:6px;align-items:center;margin:3px 0;font-variant-numeric:tabular-nums}
.bar{height:12px;background:var(--barbg);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--bar)}
.row.pick .bar i{background:var(--pick)}.row.pick{font-weight:600}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:6px 0 10px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}.chip{border-radius:999px;padding:2px 8px;color:#fff;font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;font-size:12px}.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;background:var(--c)}
h2{font-size:13px;margin:10px 0 4px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
</style></head><body>
<header><h1 id="title"></h1><span class="score" id="score"></span><span class="muted" id="clock"></span><span class="muted" id="cmd"></span></header>
<main>
<section class="card"><canvas id="pitch" width="1080" height="740"></canvas>
<div class="controls"><button id="play">Play</button><input id="scrub" type="range" min="0" value="0">
<select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div>
<div class="legend" id="legend"></div></section>
<section class="card"><h2>OpenJev decision</h2><div class="kv" id="dec"></div><h2>Split probabilities</h2><div id="bars"></div>
<h2>Our robots</h2><div class="chips" id="chips"></div></section>
</main>
<script>
const D = __PAYLOAD__;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const N = D.ts.length, F = D.field, cv = $('pitch'), cx = cv.getContext('2d');
const oj = D.openjev_is_yellow ? 'yellow' : 'blue', opp = D.openjev_is_yellow ? 'blue' : 'yellow';
$('title').textContent = D.title.replace(/_/g, ' ');
$('scrub').max = N - 1;
$('legend').innerHTML = D.tactics.map(t => `<span style="--c:var(--t-${t})">${t}</span>`).join('');
const pad = 0.35, W = 2 * (F.hl + pad), H = 2 * (F.hw + pad);
function fit(){ const s = cv.width / W; cv.height = Math.round(H * s); return s; }
let S = fit();
const X = x => (x + F.hl + pad) * S, Y = y => (F.hw + pad - y) * S;
function decisionAt(t){ let lo = 0, hi = D.decisions.length - 1, r = -1;
  while (lo <= hi){ const m = (lo + hi) >> 1; if (D.decisions[m].t <= t){ r = m; lo = m + 1; } else hi = m - 1; } return r; }
// Robots held by a committed tactic are absent from that tick's partition, so carry the last known slot forward.
const TAC = []; { let m = {}; for (const d of D.decisions){ m = {...m};
  for (const [t, ids] of Object.entries(d.partition)) for (const id of ids) m[id] = t; TAC.push(m); } }
const tacticsAt = k => TAC[k];
function scoreAt(t){ let s = [0, 0]; for (const [st, y, b] of D.scores) if (st <= t) s = [y, b]; return s; }
function drawPitch(){
  cx.fillStyle = css('--pitch'); cx.fillRect(0, 0, cv.width, cv.height);
  cx.strokeStyle = css('--pitchline'); cx.lineWidth = 2;
  cx.strokeRect(X(-F.hl), Y(F.hw), 2 * F.hl * S, 2 * F.hw * S);
  cx.beginPath(); cx.moveTo(X(0), Y(F.hw)); cx.lineTo(X(0), Y(-F.hw)); cx.stroke();
  cx.beginPath(); cx.arc(X(0), Y(0), F.cc * S, 0, 7); cx.stroke();
  for (const sg of [-1, 1]){ cx.strokeRect(X(sg > 0 ? F.hl - F.dd : -F.hl), Y(F.dw), F.dd * S, 2 * F.dw * S);
    cx.strokeRect(X(sg > 0 ? F.hl : -F.hl - 0.18), Y(F.gw), 0.18 * S, 2 * F.gw * S); }
}
function robot(p, o, color, label, ring){
  const r = 0.09 * S; cx.fillStyle = color; cx.beginPath(); cx.arc(X(p[0]), Y(p[1]), r, 0, 7); cx.fill();
  if (ring){ cx.strokeStyle = ring; cx.lineWidth = 4; cx.beginPath(); cx.arc(X(p[0]), Y(p[1]), r + 3, 0, 7); cx.stroke(); }
  cx.strokeStyle = '#111'; cx.lineWidth = 2; cx.beginPath(); cx.moveTo(X(p[0]), Y(p[1]));
  cx.lineTo(X(p[0] + Math.cos(o) * 0.09), Y(p[1] + Math.sin(o) * 0.09)); cx.stroke();
  cx.fillStyle = '#111'; cx.font = `${Math.round(0.1 * S)}px system-ui`; cx.textAlign = 'center'; cx.textBaseline = 'middle';
  cx.fillText(label, X(p[0]), Y(p[1]) + r + 10);
}
function render(i){
  const t = D.ts[i], k = decisionAt(t), tac = k >= 0 ? tacticsAt(k) : {};
  drawPitch();
  D.enemy_ids.forEach((id, j) => robot(D.ep[i][j], D.eo[i][j], css('--' + opp), id));
  D.friendly_ids.forEach((id, j) => robot(D.fp[i][j], D.fo[i][j], css('--' + oj), id, id === 0 ? null : css('--t-' + (tac[id] || 'none'))));
  cx.fillStyle = css('--ball'); cx.beginPath(); cx.arc(X(D.bp[i][0]), Y(D.bp[i][1]), 0.05 * S, 0, 7); cx.fill();
  const sc = scoreAt(t), ojs = D.openjev_is_yellow ? sc[0] : sc[1], ops = D.openjev_is_yellow ? sc[1] : sc[0];
  $('score').textContent = `OpenJev (${oj}) ${ojs} – ${ops} opponent (${opp})`;
  $('clock').textContent = `t = ${t.toFixed(1)} s`; $('cmd').textContent = D.cmd[i].toLowerCase().replace(/_/g, ' ');
  const d = k >= 0 ? D.decisions[k] : null;
  $('dec').innerHTML = d ? [
    ['split', d.fallback ? 'fallback (tiki_taka)' : d.choice], ['confidence', d.confidence ?? '–'],
    ['possession', d.possession ?? '–'], ['decided at', d.t.toFixed(2) + ' s'],
    ['latency', d.cached ? 'cache hit' : (d.latency_ms ?? '–') + ' ms']].map(([a, b]) => `<span class="muted">${a}</span><span>${b}</span>`).join('')
    : '<span class="muted">no decision yet</span><span></span>';
  $('bars').innerHTML = D.split_keys.map(key => { const p = d ? (d.probs[key] || 0) : 0;
    return `<div class="row ${d && d.choice === key ? 'pick' : ''}"><span>${key}</span><span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span>${p.toFixed(2)}</span></div>`; }).join('');
  $('chips').innerHTML = D.friendly_ids.filter(id => id !== 0).map(id => { const t2 = tac[id] || 'none';
    return `<span class="chip" style="background:var(--t-${t2})">#${id} ${t2}</span>`; }).join('');
  $('scrub').value = i;
}
let i = 0, playing = false, last = 0;
function loop(now){ if (!playing) return; const dt = (now - last) / 1000; last = now;
  const target = D.ts[i] + dt * +$('speed').value; while (i < N - 1 && D.ts[i + 1] <= target) i++;
  render(i); if (i >= N - 1){ playing = false; $('play').textContent = 'Play'; } else requestAnimationFrame(loop); }
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play';
  if (playing){ if (i >= N - 1) i = 0; last = performance.now(); requestAnimationFrame(loop); } };
$('scrub').oninput = e => { i = +e.target.value; render(i); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(i));
render(0);
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("npz", type=Path, help="columnar replay (.npz)")
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help="OpenJev decision log (default: sibling .openjev.jsonl)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output HTML (default: next to the replay)",
    )
    ap.add_argument(
        "--stride", type=int, default=2, help="keep every Nth tick (2 = 30 fps)"
    )
    args = ap.parse_args()

    log = args.log or args.npz.with_name(
        args.npz.name.replace(".npz", ".openjev.jsonl")
    )
    payload = build_payload(args.npz, log, args.stride)
    out = args.out or args.npz.with_suffix(".html")
    out.write_text(
        PAGE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    )
    print(
        f"wrote {out}  ({len(payload['ts'])} frames, {len(payload['decisions'])} decisions)"
    )


if __name__ == "__main__":
    main()
