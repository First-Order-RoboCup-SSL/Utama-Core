"""`tools/convert_ssl_log.py`'s per-log corpus job: what it downloads, converts and keeps."""

from __future__ import annotations

import json

from tools import convert_ssl_log


def _job(tmp_path, keep_raw):
    return {
        "name": "2024-07-01_10-00_A-vs-B.log.gz",
        "url": "https://example.invalid/log",
        "bytes": 3,  # the fake download, b"log"
        "year": 2024,
        "out": str(tmp_path / "out"),
        "raw_dir": str(tmp_path / "raw"),
        "raw_vision": False,
        "backfill_extras": True,
        "keep_raw": keep_raw,
    }


def _fakes(monkeypatch):
    calls = {"download": 0, "convert": 0}

    def _download(_url, path, _expected_bytes=0):
        calls["download"] += 1
        path.write_bytes(b"log")  # 3 bytes

    def _convert(_raw, out_dir, stem, raw_vision, write_replay):
        calls["convert"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.npz").write_bytes(b"")
        (out_dir / f"{stem}.source.json").write_text(json.dumps({"extras": {"raw_vision": raw_vision}}))
        return {}

    monkeypatch.setattr(convert_ssl_log, "_download", _download)
    monkeypatch.setattr(convert_ssl_log, "convert_log", _convert)
    monkeypatch.setattr(convert_ssl_log, "_summary_line", lambda stem, _summary: stem)
    return calls


def test_without_keep_raw_the_download_is_deleted_after_converting(tmp_path, monkeypatch):
    calls = _fakes(monkeypatch)

    assert convert_ssl_log._corpus_job(_job(tmp_path, keep_raw=False))["status"] == "converted"

    assert calls == {"download": 1, "convert": 1}
    assert not list((tmp_path / "raw").rglob("*.log.gz"))


def test_keep_raw_keeps_the_download_per_year_and_reuses_it(tmp_path, monkeypatch):
    calls = _fakes(monkeypatch)
    job = _job(tmp_path, keep_raw=True)

    assert convert_ssl_log._corpus_job(job)["status"] == "converted"
    assert (tmp_path / "raw" / "2024" / "2024-07-01_10-00_A-vs-B.log.gz").exists()
    assert convert_ssl_log._corpus_job(job)["status"] == "exists"

    assert calls == {"download": 1, "convert": 1}


def test_keep_raw_downloads_only_the_raw_log_of_an_otherwise_complete_match(tmp_path, monkeypatch):
    calls = _fakes(monkeypatch)
    convert_ssl_log._corpus_job(_job(tmp_path, keep_raw=False))

    assert convert_ssl_log._corpus_job(_job(tmp_path, keep_raw=True))["status"] == "raw kept"

    assert calls == {"download": 2, "convert": 1}
    assert (tmp_path / "raw" / "2024" / "2024-07-01_10-00_A-vs-B.log.gz").exists()


def test_keep_raw_reuses_a_download_left_by_an_interrupted_run(tmp_path, monkeypatch):
    calls = _fakes(monkeypatch)
    raw = tmp_path / "raw" / "2024" / "2024-07-01_10-00_A-vs-B.log.gz"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"log")

    assert convert_ssl_log._corpus_job(_job(tmp_path, keep_raw=True))["status"] == "converted"

    assert calls == {"download": 0, "convert": 1}


def test_limit_memory_turns_an_oversized_allocation_into_memory_error():
    # in a child process: the limit cannot be raised again once lowered
    import subprocess
    import sys

    code = (
        "from tools.convert_ssl_log import limit_memory\n"
        "limit_memory(1.5)\n"
        "try:\n"
        "    bytearray(2 * 2**30)\n"
        "except MemoryError:\n"
        "    print('MemoryError')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "MemoryError"


class _Response:
    def __init__(self, body, status):
        self._body, self.status = body, status

    def read(self, n=-1):
        chunk, self._body = (self._body, b"") if n < 0 else (self._body[:n], self._body[n:])
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_a_download_cut_short_resumes_from_where_it_stopped(tmp_path, monkeypatch):
    body = b"0123456789"
    requests = []

    def _urlopen(request, timeout):
        start = int(request.get_header("Range", "bytes=0-")[6:-1])
        requests.append(start)
        if start == 0:
            return _Response(body[:6], 200)  # the share closes the connection early, no error
        return _Response(body[start:], 206)

    monkeypatch.setattr(convert_ssl_log.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(convert_ssl_log.time, "sleep", lambda _s: None)
    path = tmp_path / "log.gz"

    convert_ssl_log._download("https://example.invalid/log", path, expected_bytes=len(body))

    assert path.read_bytes() == body
    assert requests == [0, 6]


def test_a_kept_download_of_the_wrong_size_is_fetched_again(tmp_path, monkeypatch):
    calls = _fakes(monkeypatch)
    job = {**_job(tmp_path, keep_raw=True), "bytes": 3}
    raw = tmp_path / "raw" / "2024" / "2024-07-01_10-00_A-vs-B.log.gz"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"lo")  # cut short by an earlier run

    assert convert_ssl_log._corpus_job(job)["status"] == "converted"
    assert calls["download"] == 1
