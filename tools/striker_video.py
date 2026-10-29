"""striker_video.py — side-by-side videos of the 1v1 duel (`striker_duel.py`) or the ball chase
(`chase_e2e.py`): the same starts played by each of `--players`, one panel each, captioned with who
plays and how the episode ended, after a title card. Plays in rsim, records a replay
(`replays/striker_video/<task>.npz`), renders with `utama_core.analysis.render_clip` (needs ffmpeg).

  pixi run python -m tools.striker_video duel --players rule,replays/striker/rl2/w150.npz,replays/striker/e2e2/e195.npz
  pixi run python -m tools.striker_video chase --players go_to_ball,replays/chase/c1/e300.npz --starts 8

A player is `rule` (duel) or `go_to_ball`/`approach` (chase), or a network's weights: the duel tells a
high-level network (`striker_rl.py`) from an end-to-end one (`striker_e2e_rl.py`) by its arrays. Starts
are the evaluation's (`--seed 9999`), in order: `--starts N` takes the first N.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

from tools import chase_e2e, striker_duel, striker_e2e
from utama_core.analysis.render_clip import render_clip
from utama_core.config.settings import REPLAY_BASE_PATH

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# render_clip's 1280x800 full-pitch frame below its score strip: the attacking half for the duel.
CROP = {"duel": "760:752:458:48", "chase": "1280:752:0:48"}
PANEL_W = {"duel": 608, "chase": 940}
RESULT = {"goal": "GOAL", "caught": "CAUGHT", "saved": "SAVED", "timeout": "TIMEOUT"}
RESULT.update({"stopped:out_of_bounds": "OUT"})
SUCCESS = {"goal", "caught"}


def _text(s: str, x: str, y: str, size: int, color: str = "white") -> str:
    s = s.replace(":", r"\:").replace("'", "")
    return (
        f"drawtext=fontfile={FONT}:text='{s}':x={x}:y={y}:fontsize={size}:fontcolor={color}"
        ":box=1:boxcolor=black@0.55:boxborderw=8:expansion=none"
    )


def _ffmpeg(*args: str) -> None:
    subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", *args], check=True)


def label(player: str) -> str:
    if not player.endswith(".npz"):
        return {"rule": "Rule (flat_rule's shot)", "go_to_ball": "go_to_ball skill", "approach": "approach skill"}[
            player
        ]
    params = striker_duel.load_params(Path(player))
    kind = "end-to-end" if "log_std" in params else "high-level"
    return f"RL {kind} ({Path(player).parent.name}/{Path(player).stem})"


def tactic(task: str, player: str, seed: int):
    rng = np.random.default_rng(seed)
    if task == "chase":
        if player.endswith(".npz"):
            return chase_e2e.ChaseTactic(striker_e2e.actor(striker_duel.load_params(Path(player)), True, rng))
        return chase_e2e.SkillChase(player)
    if player == "rule":
        return striker_duel.StrikerTactic(None)
    params = striker_duel.load_params(Path(player))
    if "log_std" in params:
        return striker_e2e.E2ETactic(act=striker_e2e.actor(params, True, rng))
    return striker_duel.StrikerTactic(striker_duel.net_chooser(params, True, rng))


def play(task: str, players: list[str], starts: int, seed: int) -> tuple[Path, list[list[dict]]]:
    """Every player on every start, in one recorded match; [start][player] -> the episode."""
    name = f"striker_video/{task}"
    env = chase_e2e.Chase(record=name) if task == "chase" else striker_duel.Duel(record=name)
    rows = []
    for i in range(starts):
        row = []
        for player in players:
            s = seed * 1_000_000 + i
            t = tactic(task, player, s)
            keeper = ["default", "flat"][i % 2]
            outcome = env.episode(random.Random(s), t) if task == "chase" else env.episode(random.Random(s), t, keeper)
            t1 = env.runner.my.current_game_frame.ts
            row.append({"player": player, "start": i, "keeper": keeper, "t0": t.t0, "t1": t1, "outcome": outcome})
            print(row[-1], flush=True)
        rows.append(row)
    env.runner.close()
    return REPLAY_BASE_PATH / f"{name}.npz", rows


def panel(task: str, replay: Path, ep: dict, pad_to: float, work: Path, k: int) -> Path:
    raw = work / f"start{ep['start']}_player{k}_raw.mp4"
    # A goal's last tick already has the ball teleported for the kickoff: end a tick before it.
    end = ep["t1"] - (0.02 if ep["outcome"] == "goal" else -0.02)
    render_clip(replay, ep["t0"], end, raw, camera="full")
    length = ep["t1"] - ep["t0"]
    result = RESULT.get(ep["outcome"], ep["outcome"].replace("stopped:", "FOUL: ").upper())
    if ep["outcome"] == "caught":
        result = f"CAUGHT {length:.2f} s"
    against = f"vs {'default keeper' if ep['keeper'] == 'default' else 'flat_rules keeper'}  " if task == "duel" else ""
    vf = ",".join(
        [
            f"crop={CROP[task]}",
            f"tpad=start_duration=0.6:start_mode=clone:stop_duration={1.5 + max(0.0, pad_to - length):.2f}:stop_mode=clone",
            _text(label(ep["player"]), "16", "16", 36),
            _text(f"{against}(start #{ep['start']})", "16", "70", 27),
            _text(result, "(w-tw)/2", "h*0.42", 72, "lime" if ep["outcome"] in SUCCESS else "orange")
            + f":enable='gte(t,{0.6 + length:.2f})'",
        ]
    )
    out = raw.with_name(raw.name.replace("_raw", ""))
    _ffmpeg("-i", str(raw), "-vf", vf, "-r", "60", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out))
    return out


def title(lines: list[str], work: Path) -> Path:
    out = work / "title.mp4"
    top = -(52 + 34 * 1.9 * (len(lines) - 1)) / 2
    ys = [top] + [top + 52 * 1.9 + 34 * 1.9 * (k - 1) for k in range(1, len(lines))]
    draws = [
        _text(line, "(w-tw)/2", f"(h/2){y:+.0f}", 52 if k == 0 else 34) for k, (line, y) in enumerate(zip(lines, ys))
    ]
    color = "color=c=0x1b2a1b:s=1920x1080:d=4:r=60"
    _ffmpeg("-f", "lavfi", "-i", color, "-vf", ",".join(draws), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=("duel", "chase"))
    parser.add_argument("--players", required=True, help="comma-separated: rule, go_to_ball, approach or weights")
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--out", type=Path, help="the MP4; default replays/striker_video/<task>.mp4")
    a = parser.parse_args()
    players = a.players.split(",")
    out = a.out or REPLAY_BASE_PATH / "striker_video" / f"{a.task}.mp4"
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    replay, rows = play(a.task, players, a.starts, a.seed)
    wins = {p: sum(row[k]["outcome"] in SUCCESS for row in rows) for k, p in enumerate(players)}
    what = "1v1 duel: attacker vs goalkeeper" if a.task == "duel" else "Ball chase: one robot gets the ball"
    side = ["left", "middle", "right"] if len(players) == 3 else ["left", "right"] if len(players) == 2 else []
    names = [f"{side[k]}: " if side else "" for k in range(len(players))]
    lines = [what, *(f"{names[k]}{label(p)}  -  {wins[p]}/{a.starts}" for k, p in enumerate(players))]
    lines.append(f"starts #0-{a.starts - 1} of the evaluation set, rsim")
    clips = [title(lines, work)]
    w = min(PANEL_W[a.task], (1880 // len(players)) // 2 * 2)
    h = round(w * 752 / int(CROP[a.task].split(":")[0]) / 2) * 2
    for row in rows:
        longest = max(e["t1"] - e["t0"] for e in row)
        panels = [panel(a.task, replay, e, longest, work, k) for k, e in enumerate(row)]
        graph = "".join(f"[{k}:v]scale={w}:{h},setpts=PTS-STARTPTS[v{k}];" for k in range(len(panels)))
        x = (1920 - w * len(panels)) // 2
        graph += "".join(f"[v{k}]" for k in range(len(panels))) + f"hstack=inputs={len(panels)},"
        graph += f"pad=1920:1080:{x}:{(1080 - h) // 2}:color=0x1b2a1b"
        clip = work / f"start_{row[0]['start']}.mp4"
        inputs = [arg for p in panels for arg in ("-i", str(p))]
        _ffmpeg(*inputs, "-filter_complex", graph, "-r", "60", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip))
        clips.append(clip)
    listing = work / "clips.txt"
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    _ffmpeg(
        "-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "60", str(out)
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
