#!/usr/bin/env python3
# ruff: noqa: E402
"""One match window as swimlanes: referee command, each robot's tactic, distance to the ball, events.

    pixi run python tools/match_timeline.py replays/tournament_<id>/<a>_vs_<b>.npz 100 160 --side friendly

Stacked panels on one time axis (Gantt-style, no dual axes):
1. the referee command band: live play, STOP, kickoff, free kick, penalty, placement, HALT (goal and
   timeout stoppages count as STOP), labelled;
2. one lane per robot of the chosen side: a span per tactic the robot was in (the Tactic class named in
   the `.intentions.jsonl` row beside the replay), a thin black tick at each switch to another tactic,
   and a dark strip along the bottom edge where the replay says the robot had the ball;
3. the distance from each of the side's robots (up to 6) to the ball, one colour per robot id;
4. events along the top: goals, fouls (rule names) and shots, ours in blue and theirs in orange.

The intentions log is written for the friendly side only (config_a), so `--side enemy` leaves the
lanes empty; the command band, possession strips and distances still work.

Prints, for the side: tactic switches per minute and the longest single tactic span, per robot.
`--out` defaults to a `.timeline_<side>_<t0>-<t1>.png` beside the replay.

Reuses `tools/replay_trace.py` for the event list (kicks, possession, referee changes, fouls, goals) and
the status messages, and `analysis/turnover_breakdown.analyse_match` for shots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import replay_trace  # noqa: E402
from utama_core.analysis import turnover_breakdown  # noqa: E402
from utama_core.analysis.findings import Finding, write_jsonl  # noqa: E402
from utama_core.entities.referee.referee_command import RefereeCommand  # noqa: E402

# A flip-back is a tactic held for less than FLIP_S between two spans of the same other tactic (A, B, A):
# the robot reverted before B could do anything. A robot with at least FLIP_BACKS_PER_MIN of them per minute
# of live play gets a `tactic_thrash` finding. Raw switch rate is not the signal: most strategies switch
# 10-25 times per live minute by design (attack/defence on possession). Measured on tournament_20261017_151052
# (66 matches, friendly robots, median flip-backs under 1 s per live min): give_and_go_solo 0.0, decoy_and_overload
# 0.1, high_press 0.3, overload_press 0.5; counter_flow, clear_danger, clear_press_plus, split_shape,
# press_trigger_flow, score_aware_counter_flow 4.6-5.6; press_and_pass 13.4. 2 per min (one every 30 s of live
# play) sits between the steady and the flipping strategies.
FLIP_S = 1.0
FLIP_BACKS_PER_MIN = 2.0
# Under this much live play a rate is noise (one switch in 10 s reads as 6 per minute): no thrash finding.
MIN_LIVE_S = 30.0

# Styling copied from tools/signal_report.py so the figures read as one set.
INK, INK_2, INK_3, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
FOR, AGAINST = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER = "#b9b8b3"  # tactics beyond the seven slots fold into this grey
UNASSIGNED_EDGE = "#c9c8c3"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "font.size": 9,
    }
)

# Referee command groups, in the order the legend lists them. Pastel fills; text is drawn on top.
REF_GROUP_FILL = {
    "live play": GRID,
    "STOP": INK_3,
    "HALT": INK_2,
    "kickoff": "#c9d7ee",
    "free kick": "#f6d7c3",
    "penalty": "#eacbdc",
    "placement": "#efe6b8",
    "other": "#fcfcfb",
    "no referee": "#fcfcfb",
}
REF_GROUP_TEXT = {"STOP": SURFACE, "HALT": SURFACE}

NO_TACTIC = None  # a Span label for "no tactic assigned"
BALL = "ball"  # a possession Span label


class Span(NamedTuple):
    t0: float
    t1: float
    label: Optional[str]


# --- intentions -> per-robot tactic spans -------------------------------------------------------


def load_intentions(npz_path: Path) -> list[dict]:
    """The `intention` rows of the `.intentions.jsonl` beside the replay (empty if there is none)."""
    path = npz_path.with_name(npz_path.name[: -len(".npz")] + ".intentions.jsonl")
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") == "intention":
                rows.append(row)
    return sorted(rows, key=lambda r: r["sim_time"])


def goalkeeper_ids(npz_path: Path) -> set[int]:
    """Robot ids the strategy marked as keeper (`highlights.goalkeeper` trace rows). The keeper has no slot."""
    path = npz_path.with_name(npz_path.name[: -len(".npz")] + ".intentions.jsonl")
    if not path.exists():
        return set()
    keepers: set[int] = set()
    with open(path) as f:
        for line in f:
            if '"highlights.goalkeeper"' in line:
                row = json.loads(line)
                keepers |= {int(k) for k in (row.get("value") or {})}
    return keepers


def tactic_changes(rows: list[dict], robot_ids) -> dict[int, list[tuple[float, Optional[str]]]]:
    """For each robot: the (sim_time, tactic) points where its tactic changes, in time order.

    A robot's tactic is the slot (`tactic_id`) whose latest row lists it. The name is the Tactic class in
    the row's `note`; rows whose note is not a class name (deadline releases, "released ...") keep the name
    the slot had. A "referee reset" row clears every slot.
    """
    members: dict[str, frozenset] = {}  # slot -> robots, insertion order = last update last
    names: dict[str, str] = {}  # slot -> Tactic class name
    changes: dict[int, list[tuple[float, Optional[str]]]] = {int(r): [] for r in robot_ids}
    for row in rows:
        t = float(row["sim_time"])
        if row["tactic_id"] == "referee reset":
            members.clear()
        else:
            slot = row["tactic_id"]
            note = row.get("note") or ""
            if note.isidentifier():
                names[slot] = note
            names.setdefault(slot, slot)
            members.pop(slot, None)
            members[slot] = frozenset(int(r) for r in row["robot_ids"])
        for rid in changes:
            label = None
            for slot in reversed(list(members)):
                if rid in members[slot]:
                    label = names[slot]
                    break
            points = changes[rid]
            if not points and label is None:
                continue
            if points and points[-1][1] == label:
                continue
            if points and points[-1][0] == t:
                points[-1] = (t, label)  # several rows at one tick: the last one wins
            else:
                points.append((t, label))
    return changes


def spans_from_changes(changes: list[tuple[float, Optional[str]]], t0: float, t1: float) -> list[Span]:
    """Spans tiling [t0, t1] from change points; before the first change a robot has no tactic."""
    points = [(-math.inf, NO_TACTIC)] + list(changes)
    out: list[Span] = []
    for i, (start, label) in enumerate(points):
        end = points[i + 1][0] if i + 1 < len(points) else math.inf
        a, b = max(start, t0), min(end, t1)
        if b <= a:
            continue
        if out and out[-1].label == label:
            out[-1] = Span(out[-1].t0, b, label)
        else:
            out.append(Span(a, b, label))
    return out


def runs(flag: np.ndarray, t: np.ndarray, label: str) -> list[Span]:
    """Contiguous True runs of `flag` over the sample times `t` (each run ends at the next sample)."""
    out: list[Span] = []
    start = None
    for i, on in enumerate(flag):
        if on and start is None:
            start = t[i]
        elif not on and start is not None:
            out.append(Span(float(start), float(t[i]), label))
            start = None
    if start is not None and float(t[-1]) > float(start):
        out.append(Span(float(start), float(t[-1]), label))
    return out


def switch_times(spans: list[Span]) -> list[float]:
    """Start times of spans whose tactic differs from the previous assigned tactic (unassigned gaps skipped)."""
    out = []
    prev = None
    for s in spans:
        if s.label is None:
            continue
        if prev is not None and s.label != prev:
            out.append(s.t0)
        prev = s.label
    return out


def lane_summary(spans: list[Span], t0: float, t1: float) -> dict:
    assigned = [s for s in spans if s.label is not None]
    longest = max(assigned, key=lambda s: s.t1 - s.t0, default=None)
    minutes = max(t1 - t0, 1e-9) / 60.0
    switches = len(switch_times(spans))
    return {
        "switches": switches,
        "per_min": switches / minutes,
        "longest": None if longest is None else (longest.label, longest.t1 - longest.t0),
    }


def tactic_colours(lanes: list[list[Span]]) -> dict[str, str]:
    """Slot colours for the window's tactics, most time first; tactics beyond the seven slots are OTHER."""
    total: Counter = Counter()
    for spans in lanes:
        for s in spans:
            if s.label is not None:
                total[s.label] += s.t1 - s.t0
    ranked = [name for name, _ in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {name: SLOTS[i] for i, name in enumerate(ranked[: len(SLOTS)])}


# --- referee command band -----------------------------------------------------------------------


def ref_group(command: Optional[str]) -> str:
    """The legend group of a referee command name ("-" or None when the replay has no referee)."""
    if command in (None, "-"):
        return "no referee"
    if command in ("NORMAL_START", "FORCE_START"):
        return "live play"
    if command == "HALT":
        return "HALT"
    if command == "STOP" or command.startswith(("GOAL_", "TIMEOUT_")):
        return "STOP"
    if command.startswith("PREPARE_KICKOFF"):
        return "kickoff"
    if command.startswith(("DIRECT_FREE", "INDIRECT_FREE")):
        return "free kick"
    if command.startswith("PREPARE_PENALTY"):
        return "penalty"
    if command.startswith("BALL_PLACEMENT"):
        return "placement"
    return "other"


def group_spans(groups: list[str], t: np.ndarray, t0: float, t1: float) -> list[Span]:
    """Runs of one referee group over the sample times `t`, clipped to [t0, t1]."""
    out: list[Span] = []
    for i, g in enumerate(groups):
        a = float(t[i])
        b = float(t[i + 1]) if i + 1 < len(t) else t1  # the last sample lasts to the window end
        a, b = max(a, t0), min(b, t1)
        if b <= a:
            continue
        if out and out[-1].label == g:
            out[-1] = Span(out[-1].t0, b, g)
        else:
            out.append(Span(a, b, g))
    return out


# --- drawing ------------------------------------------------------------------------------------

_LANE_H = 0.8
_LABEL_CHAR_IN = 0.062  # width of one 7 pt character, inches (approximate)
_AXIS_IN = 10.6  # width of the plotting area, inches


def _fits(text: str, span_s: float, seconds_per_inch: float) -> bool:
    return len(text) * _LABEL_CHAR_IN * seconds_per_inch <= span_s


def _ink_on(fill: str) -> str:
    r, g, b = (int(fill[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.55 else SURFACE


LEGEND_COLS = 6


def _legend_handles(colours: dict[str, str], lanes: list[list[Span]], ref_spans: list[Span], events: list[dict]):
    """Every key the figure uses, once: tactic colours, the other/none marks, possession, referee, events."""
    handles = [Patch(facecolor=fill, edgecolor=fill, label=name) for name, fill in colours.items()]
    all_labels = {s.label for spans in lanes for s in spans}
    if any(lab is not None and lab not in colours for lab in all_labels):
        handles.append(Patch(facecolor=OTHER, label="other tactic"))
    if None in all_labels:
        handles.append(Patch(facecolor="none", edgecolor=UNASSIGNED_EDGE, hatch="////", label="no tactic"))
    handles.append(Patch(facecolor=INK, label="had the ball"))
    handles.append(plt.Line2D([], [], color=INK, lw=1.0, label="switched tactic"))
    handles += [Patch(facecolor=REF_GROUP_FILL[g], edgecolor=INK_3, lw=0.4, label=g) for g in REF_GROUP_FILL]
    kinds = {e["kind"] for e in events}
    if "GOAL" in kinds:
        handles.append(
            plt.Line2D([], [], marker="D", ls="", color=FOR, mec=INK, label="goal (blue ours, orange theirs)")
        )
    if "FOUL" in kinds:
        handles.append(plt.Line2D([], [], marker="v", ls="", color=FOR, mec=INK, label="foul (ours)"))
        handles.append(plt.Line2D([], [], marker="v", ls="", color=AGAINST, mec=INK, label="foul (theirs)"))
    if "SHOT" in kinds:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=FOR, mec=INK, label="shot (ours / theirs)"))
    return handles


def render_timeline(
    out: Path,
    *,
    t0: float,
    t1: float,
    side: str,
    robot_ids: list[int],
    lanes: list[list[Span]],
    possession: list[list[Span]],
    ref_spans: list[Span],
    t: np.ndarray,
    dist: np.ndarray,
    events: list[dict],
    keepers: set[int] = frozenset(),
    title: str = "",
) -> Path:
    """Draw the swimlanes to `out` (PNG). `dist` is (len(t), len(robot_ids)) metres to the ball."""
    k = len(robot_ids)
    colours = tactic_colours(lanes)
    sec_per_in = (t1 - t0) / _AXIS_IN
    height_ratios = [1.3, 1.0] + [1.0] * k + [3.2]
    handles = _legend_handles(colours, lanes, ref_spans, events)
    legend_rows = math.ceil(len(handles) / LEGEND_COLS)
    legend_h = 0.2 * legend_rows + 0.15  # inches, under the panels
    fig_h = 0.9 + 0.36 * k + 0.5 + 1.2 + legend_h
    fig = plt.figure(figsize=(12, fig_h))
    gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.12)
    ax_ev = fig.add_subplot(gs[0])
    ax_ref = fig.add_subplot(gs[1], sharex=ax_ev)
    lane_axes = [fig.add_subplot(gs[2 + i], sharex=ax_ev) for i in range(k)]
    ax_d = fig.add_subplot(gs[2 + k], sharex=ax_ev)

    # 1. events along the top
    ax_ev.set_ylim(0, 1)
    ax_ev.set_yticks([])
    ax_ev.spines["left"].set_visible(False)
    ax_ev.spines["bottom"].set_visible(False)
    ax_ev.tick_params(axis="x", bottom=False, labelbottom=False)
    alt = 0
    for ev in events:
        colour = FOR if ev.get("side") == "ours" else AGAINST
        if ev["kind"] == "GOAL":
            for ax in lane_axes + [ax_d]:
                ax.axvline(ev["t"], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
            ax_ev.plot([ev["t"]], [0.5], marker="D", ms=5, color=colour, mec=INK, mew=0.6, zorder=3)
            ax_ev.text(ev["t"], 0.98, ev["text"], fontsize=7, ha="center", va="top", color=INK, fontweight="bold")
        elif ev["kind"] == "FOUL":
            ax_ev.plot([ev["t"]], [0.5], marker="v", ms=6, color=colour, mec=INK, mew=0.4, zorder=3)
            y = 0.3 if alt % 2 else 0.02
            alt += 1
            ax_ev.text(
                ev["t"] + 0.2 * sec_per_in, y, ev["rule"].replace("_", " "), fontsize=6.5, color=INK_2, va="bottom"
            )
        elif ev["kind"] == "SHOT":
            ax_ev.plot([ev["t"]], [0.5], marker="o", ms=4, color=colour, mec=INK, mew=0.4, zorder=3)
    ax_ev.set_ylabel("events", rotation=0, ha="right", va="center", color=INK_2, fontsize=8)
    if title:
        ax_ev.set_title(title, loc="left", fontsize=10)

    # 2. referee command band
    ax_ref.set_ylim(0, 1)
    ax_ref.set_yticks([])
    ax_ref.spines["left"].set_visible(False)
    ax_ref.tick_params(axis="x", bottom=False, labelbottom=False)
    for s in ref_spans:
        fill = REF_GROUP_FILL.get(s.label, OTHER)
        ax_ref.broken_barh([(s.t0, s.t1 - s.t0)], (0.08, 0.84), facecolors=fill, edgecolors=SURFACE, linewidth=0.5)
        if _fits(s.label, s.t1 - s.t0, sec_per_in):
            ax_ref.text(
                (s.t0 + s.t1) / 2,
                0.5,
                s.label,
                fontsize=7,
                ha="center",
                va="center",
                color=REF_GROUP_TEXT.get(s.label, INK),
            )
    ax_ref.set_ylabel("referee", rotation=0, ha="right", va="center", color=INK_2, fontsize=8)

    # 3. one lane per robot
    for i, rid in enumerate(robot_ids):
        ax = lane_axes[i]
        y = 0.0
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for s in lanes[i]:
            if s.label is None:
                ax.broken_barh(
                    [(s.t0, s.t1 - s.t0)],
                    (y - _LANE_H / 2, _LANE_H),
                    facecolors="none",
                    edgecolors=UNASSIGNED_EDGE,
                    hatch="////",
                    linewidth=0,
                )
                continue
            fill = colours.get(s.label, OTHER)
            ax.broken_barh(
                [(s.t0, s.t1 - s.t0)],
                (y - _LANE_H / 2, _LANE_H),
                facecolors=fill,
                edgecolors=SURFACE,
                linewidth=0.5,
            )
            if _fits(s.label, s.t1 - s.t0, sec_per_in):
                ax.text(
                    (s.t0 + s.t1) / 2,
                    y,
                    s.label,
                    fontsize=7,
                    ha="center",
                    va="center",
                    color=_ink_on(fill),
                    clip_on=True,
                )
        for tt in switch_times(lanes[i]):
            ax.plot([tt, tt], [y - _LANE_H / 2, y + _LANE_H / 2], color=INK, lw=1.0, zorder=3)
        for p in possession[i]:
            ax.broken_barh([(p.t0, p.t1 - p.t0)], (y - _LANE_H / 2, 0.14), facecolors=INK, linewidth=0, zorder=4)
        ax.set_ylabel(
            f"R{rid}" + (" keeper" if rid in keepers else ""),
            rotation=0,
            ha="right",
            va="center",
            color=INK,
            fontsize=8,
        )

    # 4. distance to the ball
    ax_d.set_ylabel("dist to ball (m)", rotation=0, ha="right", va="center", color=INK_2, fontsize=8)
    ymax = float(np.nanmax(dist)) if dist.size and np.isfinite(dist).any() else 1.0
    ax_d.set_ylim(0, max(1.0, min(ymax, 12.0)) * 1.05)
    ax_d.set_xlim(t0, t1)
    ax_d.grid(axis="y", color=GRID, lw=0.6)
    end_labels = []
    for i, rid in enumerate(robot_ids):
        colour = SLOTS[i % len(SLOTS)]
        ax_d.plot(t, dist[:, i], color=colour, lw=1.1)
        finite = np.where(np.isfinite(dist[:, i]))[0]
        if finite.size:
            end_labels.append((float(dist[finite[-1], i]), rid, colour))
    # Direct labels at the right edge, nudged apart so two robots never share a line.
    last = -math.inf
    gap = 0.12 * ax_d.get_ylim()[1]  # enough for one 7 pt line of text at this panel height
    for val, rid, colour in sorted(end_labels):
        y = max(val, last + gap)
        last = y
        ax_d.text(t1, y, f" R{rid}", fontsize=7, color=colour, va="center", ha="left", clip_on=False)
    ax_d.set_xlabel("time in window (s)", color=INK_2)
    ax_d.set_xlim(t0, t1)

    # Legend, once, under the figure.
    fig.legend(
        handles=handles, loc="lower center", ncol=LEGEND_COLS, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0)
    )
    fig.subplots_adjust(left=0.1, right=0.93, top=1 - 0.45 / fig_h, bottom=(legend_h + 0.2) / fig_h)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


# --- glue ---------------------------------------------------------------------------------------


def _summary_lines(
    robot_ids: list[int], lanes: list[list[Span]], t0: float, t1: float, keepers: set[int] = frozenset()
) -> list[str]:
    lines = [f"tactic switches per minute, window {t0:.1f}-{t1:.1f} s (spans are clipped to the window):"]
    lines.append(f"  {'robot':>7} {'switches':>8} {'per min':>8}  longest tactic span")
    for rid, spans in zip(robot_ids, lanes):
        s = lane_summary(spans, t0, t1)
        longest = "-" if s["longest"] is None else f"{s['longest'][0]} {s['longest'][1]:.1f} s"
        name = f"R{rid}" + (" keeper" if rid in keepers else "")
        lines.append(f"  {name:>7} {s['switches']:>8} {s['per_min']:>8.2f}  {longest}")
    return lines


class Window(NamedTuple):
    """One window's computed data, before any drawing."""

    t0: float
    t1: float
    robot_ids: list[int]
    lanes: list[list[Span]]
    possession: list[list[Span]]
    ref_spans: list[Span]
    tw: np.ndarray  # sample times in the window, on the replay's clock from its first frame
    dist: np.ndarray  # (len(tw), robots) metres to the ball
    keepers: set[int]
    live_s: float  # seconds of live play (NORMAL_START / FORCE_START) in the window
    notes: list[str]
    z: dict  # the replay's arrays, as loaded


def load_window(npz_path: Path, t0: Optional[float], t1: Optional[float], side: str) -> Window:
    """The window's lanes, possession, referee spans and distances; no drawing and no shot analysis."""
    z = dict(np.load(npz_path))
    ts = z["ts"]
    t = ts - ts[0]
    t_end = float(t[-1])
    t0 = 0.0 if t0 is None else max(0.0, t0)
    t1 = t_end if t1 is None else min(t_end, t1)
    if t1 <= t0:
        raise SystemExit(f"empty window: t0={t0} t1={t1} (the replay runs 0-{t_end:.1f} s)")
    win = (t >= t0) & (t <= t1)
    tw = t[win]

    notes: list[str] = []
    robot_ids = [int(r) for r in z[f"{side}_ids"]]
    rows = load_intentions(npz_path) if side == "friendly" else []
    if side == "enemy":
        notes.append("note: the intentions log is written for the friendly side only; enemy lanes are empty")
    elif not rows:
        notes.append(f"note: no {npz_path.name[:-4]}.intentions.jsonl beside the replay; lanes are empty")
    changes = tactic_changes(rows, robot_ids)
    lanes = [spans_from_changes(changes[rid], t0, t1) for rid in robot_ids]
    keepers = goalkeeper_ids(npz_path) if side == "friendly" else set()
    has_ball = z[f"{side}_has_ball"][win]
    possession = [runs(has_ball[:, i], tw, BALL) for i in range(len(robot_ids))]

    cmd_names = [
        RefereeCommand(int(c)).name if h else "-" for c, h in zip(z["referee_command"][win], z["has_referee"][win])
    ]
    ref_spans = group_spans([ref_group(c) for c in cmd_names], tw, t0, t1)
    live_s = sum(s.t1 - s.t0 for s in ref_spans if s.label == "live play")

    pos = z[f"{side}_p"][win]
    ball = z["ball_p"][win]
    dist = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
    return Window(t0, t1, robot_ids, lanes, possession, ref_spans, tw, dist, keepers, live_s, notes, z)


def flip_backs(spans: list[Span], max_s: float = FLIP_S) -> list[float]:
    """Start times of the tactic spans shorter than `max_s` that sit between two spans of one other tactic
    (A, B, A; unassigned gaps skipped)."""
    assigned = [s for s in spans if s.label is not None]
    return [
        b.t0
        for a, b, c in zip(assigned, assigned[1:], assigned[2:])
        if a.label == c.label != b.label and b.t1 - b.t0 < max_s
    ]


def thrash_findings(
    w: Window,
    *,
    replay: Optional[str],
    side: str,
    image: Optional[str] = None,
    threshold: float = FLIP_BACKS_PER_MIN,
) -> list[Finding]:
    """One `tactic_thrash` finding per robot whose flip-backs (`flip_backs`) reach `threshold` per minute of
    live play.

    Flip-backs are counted over the whole window; the rate divides by the live play in it, so a stoppage does
    not dilute a robot that thrashes in open play. No finding when the window has less than `MIN_LIVE_S` of
    live play: a rate over a few seconds is noise."""
    if w.live_s < MIN_LIVE_S:
        return []
    minutes = w.live_s / 60.0
    out = []
    for rid, spans in zip(w.robot_ids, w.lanes):
        flips = flip_backs(spans)
        rate = len(flips) / minutes
        if rate >= threshold:
            out.append(
                Finding(
                    tool="match_timeline",
                    kind="tactic_thrash",
                    replay=replay,
                    side=side,
                    t0=w.t0,
                    t1=w.t1,
                    robots=(rid,),
                    value=rate,
                    threshold=threshold,
                    unit="flip-backs per live min",
                    message=(
                        f"R{rid} took a tactic for under {FLIP_S:g} s and reverted {len(flips)} times in "
                        f"{minutes:.1f} min of live play ({rate:.2f} per min, threshold {threshold:g}); "
                        f"first at t={flips[0]:.1f} s."
                    ),
                    image=image,
                )
            )
    return out


def timeline_findings(replay_path: Path, t0: Optional[float], t1: Optional[float], side: str) -> list[Finding]:
    """The match_timeline findings for one window, computed without drawing (`image` is None)."""
    replay_path = Path(replay_path)
    w = load_window(replay_path, t0, t1, side)
    return thrash_findings(w, replay=str(replay_path), side=side)


def build(npz_path: Path, t0: Optional[float], t1: Optional[float], side: str, out: Optional[Path]):
    """Compute the window's data, draw it, and return (png path, summary lines, findings with `image` set)."""
    w = load_window(npz_path, t0, t1, side)
    z, ts, t0, t1 = w.z, w.z["ts"], w.t0, w.t1
    statuses = replay_trace._statuses(npz_path, len(ts))
    fouls = replay_trace._fouls(npz_path)
    events = []
    my_yellow = bool(z["my_team_is_yellow"])
    for ev in replay_trace.events(z, statuses, fouls, t0, t1):
        if ev["kind"] == "FOUL":
            events.append({"kind": "FOUL", "t": ev["t"], "rule": ev["rule"], "side": ev["side"]})
        elif ev["kind"] == "REF" and ev["command"].startswith("GOAL_"):
            scored_yellow = ev["command"] == "GOAL_YELLOW"
            ours = scored_yellow == my_yellow
            events.append(
                {
                    "kind": "GOAL",
                    "t": ev["t"],
                    "side": "ours" if ours else "theirs",
                    "text": "GOAL ours" if ours else "GOAL theirs",
                }
            )
    for shot in turnover_breakdown.analyse_match(str(npz_path))["chances"]["shots"]:
        st = float(shot["t"]) - float(ts[0])
        if t0 <= st <= t1:
            events.append({"kind": "SHOT", "t": st, "side": "ours" if shot["side"] == "friendly" else "theirs"})
    events.sort(key=lambda e: e["t"])

    if out is None:
        out = npz_path.with_name(f"{npz_path.name[:-4]}.timeline_{side}_{int(t0)}-{int(t1)}.png")
    title = f"{npz_path.name[:-4]}  |  {side}  |  {t0:.0f}-{t1:.0f} s"
    png = render_timeline(
        out,
        t0=t0,
        t1=t1,
        side=side,
        robot_ids=w.robot_ids,
        lanes=w.lanes,
        possession=w.possession,
        ref_spans=w.ref_spans,
        t=w.tw,
        dist=w.dist,
        events=events,
        keepers=w.keepers,
        title=title,
    )
    findings = thrash_findings(w, replay=str(npz_path), side=side, image=str(png))
    return png, w.notes + _summary_lines(w.robot_ids, w.lanes, t0, t1, w.keepers), findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, help="a match's .npz (its .intentions.jsonl must sit beside it)")
    parser.add_argument("t0", type=float, nargs="?", default=None, help="window start, seconds (default 0)")
    parser.add_argument("t1", type=float, nargs="?", default=None, help="window end, seconds (default: the end)")
    parser.add_argument("--t0", dest="t0_opt", type=float, default=None, help="same as the t0 argument")
    parser.add_argument("--t1", dest="t1_opt", type=float, default=None, help="same as the t1 argument")
    parser.add_argument("--side", choices=["friendly", "enemy"], default="friendly")
    parser.add_argument("--out", type=Path, default=None, help="PNG path (default: beside the replay)")
    parser.add_argument(
        "--findings", metavar="PATH", help="also write the flagged problems as JSON lines (- for stdout)"
    )
    args = parser.parse_args()
    t0 = args.t0_opt if args.t0_opt is not None else args.t0
    t1 = args.t1_opt if args.t1_opt is not None else args.t1
    png, summary, findings = build(args.replay, t0, t1, args.side, args.out)
    human = sys.stderr if args.findings == "-" else sys.stdout  # keep stdout pure JSON for "-"
    print(f"wrote {png}", file=human)
    print("\n".join(summary), file=human)
    if args.findings is not None:
        write_jsonl(findings, args.findings)


if __name__ == "__main__":
    main()
