"""What a field test's game log records besides the packets: notes, and which code ran."""

from __future__ import annotations

import json
import subprocess

from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.replay import game_log_writer
from utama_core.replay.game_log_writer import NOTE, GameLogWriter, code_version
from utama_core.replay.ssl_log import read_records
from utama_core.run.strategy_runner import _replay_name


def test_a_note_is_a_record_in_the_log(tmp_path):
    log = GameLogWriter(tmp_path / "run.log")
    log.note("ball placed by hand")
    log.close()
    assert [(k, json.loads(d)) for _, k, d in read_records(tmp_path / "run.log")] == [
        (NOTE, {"text": "ball placed by hand"})
    ]


def test_the_code_version_is_this_checkouts_commit_or_none_without_git(monkeypatch):
    version = code_version()
    assert len(version["commit"]) == 40 and isinstance(version["dirty"], bool)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(game_log_writer.subprocess, "run", no_git)
    assert code_version() is None
    monkeypatch.setattr(
        game_log_writer.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 128, "", "not a repo")
    )
    assert code_version() is None


def test_the_replay_sits_beside_the_log():
    assert _replay_name(str(REPLAY_BASE_PATH / "field" / "20261022_213347_real.log")) == "field/20261022_213347_real"
    assert _replay_name("/somewhere/else/run.log") == "run"
