"""Tests for tools/package_logs.py: tournament and field logs packaging."""

import tarfile
from pathlib import Path

from tools.package_logs import (
    collect_field_files,
    collect_tournament_files,
    compress_log_file,
    find_latest_tournament,
    package_to_tar_gz,
)


def test_collect_tournament_files_skips_npz_and_collects_metadata(tmp_path):
    run_dir = tmp_path / "tournament_20261017_000000"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}")
    (run_dir / "match_a_vs_b.stats.json").write_text("{}")
    (run_dir / "match_a_vs_b.timeline.json").write_text("{}")
    (run_dir / "match_a_vs_b.npz").write_bytes(b"large_array")
    (run_dir / "match_a_vs_b.sparse_referee.pkl").write_bytes(b"large_pickle")
    (run_dir / "match_a_vs_b.intentions.jsonl").write_text("intention\n")

    files = collect_tournament_files(run_dir, with_intentions=False)
    file_names = {f.name for f in files}

    assert "summary.json" in file_names
    assert "match_a_vs_b.stats.json" in file_names
    assert "match_a_vs_b.timeline.json" in file_names
    assert "match_a_vs_b.npz" not in file_names
    assert "match_a_vs_b.sparse_referee.pkl" not in file_names
    assert "match_a_vs_b.intentions.jsonl" not in file_names

    with_int = collect_tournament_files(run_dir, with_intentions=True)
    assert "match_a_vs_b.intentions.jsonl" in {f.name for f in with_int}


def test_collect_field_files_skips_zero_byte_and_tmp_files(tmp_path):
    field_dir = tmp_path / "field"
    field_dir.mkdir()
    (field_dir / "valid.log").write_bytes(b"ssl_data")
    (field_dir / "valid.calibration.json").write_text("{}")
    (field_dir / "empty.log").write_bytes(b"")  # 0 bytes
    (field_dir / "unfinished.tmp.npz").write_bytes(b"tmp_data")

    files = collect_field_files(field_dir, include_replays=False)
    names = {f.name for f in files}

    assert "valid.log" in names
    assert "valid.calibration.json" in names
    assert "empty.log" not in names
    assert "unfinished.tmp.npz" not in names


def test_compress_log_file_creates_gz_and_unlinks_source(tmp_path):
    log_file = tmp_path / "sample.log"
    log_file.write_bytes(b"SSL_LOG_FILE\x00\x00\x00\x01" * 100)

    gz_file = compress_log_file(log_file)

    assert gz_file.name == "sample.log.gz"
    assert gz_file.exists()
    assert not log_file.exists()
    assert gz_file.stat().st_size > 0


def test_package_to_tar_gz_creates_valid_tar(tmp_path):
    base_dir = tmp_path / "replays"
    target_dir = base_dir / "tournament_run"
    target_dir.mkdir(parents=True)
    f1 = target_dir / "summary.json"
    f1.write_text('{"status": "ok"}')

    out_archive = base_dir / "out.tar.gz"
    package_to_tar_gz([f1], base_dir=base_dir, out_tar_gz=out_archive)

    assert out_archive.exists()
    with tarfile.open(out_archive, "r:gz") as tar:
        names = tar.getnames()
        assert "tournament_run/summary.json" in names


def test_find_latest_tournament(tmp_path):
    replays = tmp_path / "replays"
    replays.mkdir()
    (replays / "tournament_20261010_100000").mkdir()
    (replays / "tournament_20261016_200000").mkdir()

    latest = find_latest_tournament(replays)
    assert latest is not None
    assert latest.name == "tournament_20261016_200000"
