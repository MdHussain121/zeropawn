"""Unit tests for the shared terminal reporter (python/common/progress.py).

Covers: banner/header/parameters/progress/status/event/heartbeat/results/
artifacts/summary/error-block formatting, cp437 ASCII-safety, NO_COLOR and
non-TTY behavior, and the thread-safe time-based rate limiter.
"""

import threading

from python.common.progress import (
    Heartbeat,
    RateLimiter,
    artifacts_list,
    banner,
    color_enabled,
    error_block,
    event_line,
    file_record,
    format_kv_line,
    header_block,
    heartbeat_line,
    is_tty,
    parameters_table,
    results_table,
    status_line,
    summary_block,
    warn_line,
)


def _all_lines(*blocks: str) -> list:
    lines = []
    for block in blocks:
        lines.extend(block.splitlines())
    return lines


def test_reporter_outputs_are_cp437_safe():
    nasty = "╔═╗║ℹ✔✦→ café naïve Ω Δ"
    blocks = [
        banner("TITLE " + nasty, "sub " + nasty),
        header_block("t", "cmd", "wd", "c", "cfg", "ckpt", "dev", "seed " + nasty),
        parameters_table([("name" + nasty, "val" + nasty, "user")]),
        format_kv_line([("k", nasty)]),
        status_line([("k", nasty)]),
        event_line("TAG", nasty),
        heartbeat_line(nasty),
        warn_line(nasty),
        results_table([("name" + nasty, nasty)]),
        artifacts_list([("p" + nasty, 12, "abc")]),
        summary_block("SUCCESS", 1.5, 0, 0),
        error_block(nasty, nasty, nasty, nasty),
    ]
    for line in _all_lines(*blocks):
        line.encode("cp437")  # must not raise


def test_banner_shape_fixed_width():
    text = banner("my-op", "sub")
    lines = text.splitlines()
    assert lines[0] == "=" * 70
    assert lines[1] == "OPERATION: my-op"
    assert lines[-1] == "=" * 70


def test_header_block_carries_identity_fields():
    text = header_block(
        "2026-10-12 04:31:02", "python x.py", "C:\\r", "abc123", "none", "none", "cpu", "n/a"
    )
    for needle in (
        "START: 2026-10-12 04:31:02",
        "COMMAND: python x.py",
        "WORKDIR: C:\\r",
        "GIT: abc123",
        "DEVICE: cpu",
        "SEED: n/a",
    ):
        assert needle in text


def test_parameters_table_marks_source():
    text = parameters_table([("month", "2026-09", "user"), ("workers", 0, "default")])
    assert "month" in text and "2026-09 [user]" in text
    assert "workers" in text and "0 [default]" in text


def test_kv_status_event_heartbeat_shapes():
    assert format_kv_line([("a", 1), ("b", 2.5)], timestamp=1700000000.0).startswith("[")
    assert "a=1 b=2.5" in format_kv_line([("a", 1), ("b", 2.5)])
    ev = event_line("CKPT", "saved x.pt #abc", timestamp=1700000000.0)
    assert ev.startswith("[") and ">>> CKPT: saved x.pt #abc" in ev
    hb = heartbeat_line("buffer=7 games=0")
    assert "alive buffer=7 games=0" in hb
    w = warn_line("low ram")
    assert "WARNING: low ram" in w


def test_results_artifacts_summary_shapes():
    res = results_table([("kept", "5 (100.00%)"), ("loss", 9.0068)])
    assert "kept: 5 (100.00%)" in res
    art = artifacts_list([("a/b.pt", 10, "deadbeef"), ("c/d", 3, None)])
    assert "a/b.pt  10  deadbeef" in art
    assert "c/d  3  -" in art
    assert artifacts_list([]) == "ARTIFACTS: none"
    summ = summary_block("SUCCESS", 12.34, 2, 0)
    assert "STATUS: SUCCESS" in summ and "DURATION: 12.3s" in summ
    assert "WARNINGS: 2" in summ and "EXIT: 0" in summ


def test_error_block_shape_and_verbose_gate():
    text = error_block("open X failed", "bad path", "check path", "logs/y.log")
    assert "ERROR: open X failed" in text
    assert "LIKELY: bad path" in text
    assert "TRY: check path" in text
    assert "LOG: logs/y.log" in text
    assert "TRACEBACK" not in text
    verbose = error_block("w", "l", "t", "log", verbose=True, traceback_text="boom")
    assert "TRACEBACK:" in verbose and "boom" in verbose
    # verbose without traceback text adds nothing sensitive
    assert "TRACEBACK" not in error_block("w", "l", "t", "log", verbose=True)


def test_file_record_hash_and_missing(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc123")
    path, size, sha = file_record(str(target))
    assert size == 6
    import hashlib

    assert sha == hashlib.sha256(b"abc123").hexdigest()
    path2, size2, sha2 = file_record(str(tmp_path / "nope.bin"))
    assert size2 == -1 and sha2 == "missing"


def test_no_color_and_non_tty_behavior(monkeypatch):
    class Tty:
        def isatty(self):
            return True

    class Pipe:
        def isatty(self):
            return False

    assert is_tty(Tty()) is True
    assert is_tty(Pipe()) is False
    assert is_tty(None) in (True, False)  # environment-dependent, must not raise

    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color_enabled(Tty()) is True
    assert color_enabled(Pipe()) is False
    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled(Tty()) is False


def test_rate_limiter_time_gate():
    limiter = RateLimiter(10.0)
    assert limiter.should_emit(current_time=1000.0) is True
    assert limiter.should_emit(current_time=1005.0) is False
    assert limiter.should_emit(force=True, current_time=1005.0) is True
    assert limiter.should_emit(current_time=1016.0) is True
    limiter.notify(current_time=2000.0)
    assert limiter.should_emit(current_time=2005.0) is False


def test_rate_limiter_thread_safety():
    limiter = RateLimiter(10.0)
    wins = []
    lock = threading.Lock()

    def hammer():
        for _ in range(200):
            if limiter.should_emit(current_time=5000.0):
                with lock:
                    wins.append(1)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # one interval window -> exactly one winner despite 1600 racing calls
    assert wins == [1]


def test_heartbeat_silence_window():
    clock = [0.0]
    hb = Heartbeat(interval_sec=30.0, time_fn=lambda: clock[0], start_time=100.0)
    # 10 s after construction: silent
    assert hb.maybe_emit("s", current_time=110.0) is None
    hb.notify(current_time=200.0)
    assert hb.maybe_emit("s", current_time=210.0) is None
    # 31 s of silence: liveness line
    assert hb.maybe_emit("s", current_time=231.0) is not None
    # construction alone never emits immediately
    hb2 = Heartbeat(interval_sec=30.0, time_fn=lambda: clock[0], start_time=1000.0)
    assert hb2.maybe_emit("s", current_time=1000.0) is None


def test_progress_reporter_thread_safety():
    from python.common.progress import ProgressReporter

    emitted = []
    lock = threading.Lock()

    def safe_append(line):
        with lock:
            emitted.append(line)

    reporter = ProgressReporter(interval_sec=10.0, output_fn=safe_append)
    kwargs = dict(
        step=1,
        policy_loss=1.0,
        wdl_loss=1.0,
        buffer_size=1,
        games_count=1,
        would_resign_rate=0.0,
        draw_rate=0.0,
        cap_rate=0.0,
        avg_plies=1.0,
        total_sims=10,
        gpu_util=0.0,
        model_version=1,
        current_time=9000.0,
    )

    def hammer():
        for _ in range(50):
            reporter.maybe_report(**kwargs)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(emitted) == 1


def test_run_context_and_hash_helpers(tmp_path):
    from python.common.progress import git_commit, run_context, short_hash

    ctx = run_context("data-filter")
    assert ctx["operation"] == "data-filter"
    assert len(ctx["start"]) == 19  # YYYY-MM-DD HH:MM:SS
    assert isinstance(ctx["workdir"], str) and ctx["workdir"]
    assert isinstance(ctx["commit"], str) and ctx["commit"]
    assert git_commit() == ctx["commit"]

    target = tmp_path / "x.bin"
    target.write_bytes(b"data")
    import hashlib

    assert short_hash(str(target)) == hashlib.sha256(b"data").hexdigest()[:8]
    assert short_hash(str(tmp_path / "missing.bin")) == "missing"
