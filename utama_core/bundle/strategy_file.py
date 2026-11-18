"""Building, unpacking and playing `.strategy` files.

A `.strategy` file is a zip of the `utama_core` package at one commit plus a manifest naming the
strategy. It plays in its own process (`host.py`), so it keeps its own skills, tactics and motion code
while the runner around it changes: a frozen opponent, or one somebody else wrote.

It is plain Python and runs with this machine's rights. Isolate a file you did not build before
playing it.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tarfile
import weakref
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

from utama_core.bundle import wire
from utama_core.bundle.archive import SUFFIX, read_manifest, unpack
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_log import IntentionEvent, TraceEvent
from utama_core.engine.tactic import TacticTag
from utama_core.entities.data.command import RobotCommand

FORMAT_VERSION = 1
_REPO = Path(__file__).resolve().parents[2]
_OWN_CODE = Path(__file__).resolve().parent


def _wanted(path: str) -> bool:
    return path.startswith("utama_core/") and not path.startswith("utama_core/tests/") and "__pycache__" not in path


def _worktree_files() -> dict[str, bytes]:
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "utama_core"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {p: (_REPO / p).read_bytes() for p in listed if _wanted(p) and (_REPO / p).is_file()}


def _commit_files(ref: str) -> dict[str, bytes]:
    archive = subprocess.run(["git", "archive", ref, "utama_core"], cwd=_REPO, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isfile() and _wanted(m.name)}


def _commit_of(ref: str) -> str:
    # `^{commit}`: on its own an annotated tag resolves to the tag object, which is no commit.
    return subprocess.run(
        ["git", "rev-parse", f"{ref}^{{commit}}"], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def build(strategy: str, output: Path, ref: Optional[str] = None, outfield_ids: tuple = (1, 2, 3, 4, 5)) -> Path:
    """Write `output`. `ref` is a git commit; None takes the working tree as it stands.

    Same inputs, same bytes, so a file's hash identifies what it plays."""
    files = _commit_files(ref) if ref else _worktree_files()
    if f"utama_core/strategy/{strategy}.py" not in files:
        raise ValueError(f"no strategy {strategy} in utama_core/strategy/ at {ref or 'the working tree'}")

    # Git has no compiled planner or simulator, at any commit: take this machine's.
    for path in sorted((_REPO / "utama_core").rglob("*.so")):
        relative = path.relative_to(_REPO).as_posix()
        if _wanted(relative):
            files[relative] = path.read_bytes()
    # Always this version: an older commit has none.
    for path in sorted(_OWN_CODE.glob("*.py")):
        files[path.relative_to(_REPO).as_posix()] = path.read_bytes()

    manifest = {
        "format": FORMAT_VERSION,
        "strategy": strategy,
        "outfield_ids": list(outfield_ids),
        "commit": _commit_of(ref or "HEAD"),
        "working_tree": ref is None,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": f"{sys.platform}-{platform.machine()}",
    }
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    return output


def frozen(strategy: str, ref: str, directory: Path) -> Path:
    """`strategy` as it was at `ref`, built the first time and kept: a file that is never rebuilt is
    a fixed baseline, whatever happens to the planner binary or this package later."""
    path = directory / f"{strategy}@{_commit_of(ref)[:8]}{SUFFIX}"
    if not path.exists():
        partial = build(strategy, path.with_suffix(".partial"), ref)
        partial.rename(path)
    return path


def _stop(process: subprocess.Popen) -> None:
    try:
        process.stdin.close()
    except BrokenPipeError:  # it has already gone
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    process.stdout.close()


class _RemoteKernel:
    """What `StrategyRunner` reads off a kernel `Strategy` (match log, slots, partition, keeper), answered
    by the file's process, so stats, fouls, intentions and the dashboard work as for any strategy."""

    def __init__(self, owner: "BundledStrategy"):
        self._owner = owner
        self._match_log = None
        self._tactics: dict = {}

    @property
    def match_log(self):
        return self._match_log

    @match_log.setter
    def match_log(self, log) -> None:
        self._match_log = log
        if log is not None:
            self._owner._send({"type": "match_log"})

    def record(self, events: list) -> None:
        for name, fields in events:
            if name == "IntentionEvent":
                fields = {**fields, "tag": TacticTag(fields["tag"]), "robot_ids": tuple(fields["robot_ids"])}
                self._match_log._events.append(IntentionEvent(**fields))
            else:
                self._match_log._events.append(TraceEvent(**fields))

    def slot_status(self, game) -> dict:
        return self._owner._query("slot_status")

    @property
    def active_partition(self) -> dict:
        reply = self._owner._query("partition")
        # The runner names a tactic by `type(kernel._tactics[id]).__name__`.
        self._tactics = {tactic_id: type(name, (), {})() for tactic_id, name in reply["names"].items()}
        return reply["partition"]

    @property
    def _pinned_robot_id(self) -> int:
        return self._owner._goalkeeper_id


class BundledStrategy(AbstractStrategy):
    """A `.strategy` file, as `StrategyRunner` drives any strategy."""

    def __init__(self, path: str | Path, outfield_robot_ids: Optional[tuple[int, ...]] = None):
        """`outfield_robot_ids`: the robots this run has, if not the ones the file was built for
        (a field day rarely has five)."""
        super().__init__(build=None)
        self.path = Path(path)
        root = unpack(self.path)
        self.manifest = read_manifest(root)
        if self.manifest["format"] != FORMAT_VERSION:
            raise ValueError(f"{path}: format {self.manifest['format']}, this code reads {FORMAT_VERSION}")
        self._process = subprocess.Popen(
            [sys.executable, "-m", "utama_core.bundle.host", *map(str, outfield_robot_ids or ())],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self.close = weakref.finalize(self, _stop, self._process)
        self.control_scheme = self._reply()["control_scheme"]
        self._kernel_strategy = _RemoteKernel(self)
        self._sent_frame = None

    def _send(self, message: dict) -> None:
        wire.send(self._process.stdin, message)

    def _reply(self):
        reply = wire.receive(self._process.stdout)
        if reply is None:
            raise RuntimeError(f"{self.path}: its process stopped (exit code {self._process.wait()})")
        return reply

    def load_motion_controller(self, motion_controller) -> None:
        self.motion_controller = motion_controller
        self._send(
            {
                "type": "setup",
                "controller": type(motion_controller).__name__,
                "mode": motion_controller.mode.value,
                "latency_s": motion_controller.pose_ahead.latency_s,
                "robot_carry": {str(rid): list(carry) for rid, carry in motion_controller.robot_carry.items()},
                "kick_bend": {str(rid): bend for rid, bend in motion_controller.kick_bend.items()},
                "kick_bias": {str(rid): bias for rid, bias in motion_controller.kick_bias.items()},
            }
        )

    def load_game(self, game) -> None:
        super().load_game(game)
        self._sent_frame = game._current_game_standard_frame
        self._send(
            {"type": "game", "frame": wire.encode_frame(self._sent_frame), "field": wire.encode_field(game.field)}
        )

    def _sync(self) -> None:
        """The file's process sees the frame this side sees now, before it is asked anything: the
        runner reads which tactics are committed after a new frame and before stepping."""
        frame = self.game._current_game_standard_frame
        if frame is not self._sent_frame:
            self._send({"type": "frame", "frame": wire.encode_frame(frame)})
            self._sent_frame = frame

    def _query(self, name: str):
        self._sync()
        self._send({"type": "query", "name": name})
        return self._reply()

    def step(self) -> None:
        self._sync()
        self._send({"type": "tick"})
        reply = self._reply()
        self._kernel_strategy.record(reply["events"])
        commands = {int(robot_id): RobotCommand(*values) for robot_id, values in reply["commands"].items()}
        if self._kicker_charge is not None:  # needs the robot controller, which stays on this side
            commands = self._kicker_charge.apply(self.game, commands)
        for robot_id in self.game.friendly_robots:
            if robot_id in commands:
                self.robot_controller.add_robot_commands(commands[robot_id], robot_id)
        self.robot_controller.send_robot_commands()

    def debug_status(self) -> dict[int, list[str]]:
        return {int(robot_id): row for robot_id, row in self._query("debug_status").items()}
