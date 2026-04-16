"""
Plan test #21 — writer scripts must abort on .pinecone-migration-lock presence.

Covers:
- The shared guard function (abort_if_migration_in_progress) — unit tests
- scripts/add_to_vectorstore.py — subprocess test (real script entry)
- src/ingestion/process_books.py::add_books_to_vectorstore() — subprocess test
  via `python -c` wrapper

All subprocess tests use a TEMP lockfile at the real repo-root location
(src/utils/migration_lock.py::LOCKFILE) because the writers resolve that path
at runtime. Cleanup is strict — tests FAIL rather than leave a lockfile
behind that would break subsequent writer runs.

Satisfies plan AC #34.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.migration_lock import (  # noqa: E402
    ABORT_MESSAGE_PREFIX,
    LOCKFILE,
    abort_if_migration_in_progress,
)


# ---------- Unit tests on the guard function itself ----------


def test_guard_no_lockfile_returns_cleanly(tmp_path):
    absent = tmp_path / "nope.lock"
    assert not absent.exists()
    # Should return without raising / exiting
    abort_if_migration_in_progress(absent)


def test_guard_lockfile_present_exits_one(tmp_path, capsys):
    lock = tmp_path / "present.lock"
    lock.touch()
    with pytest.raises(SystemExit) as exc:
        abort_if_migration_in_progress(lock)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert ABORT_MESSAGE_PREFIX in captured.out
    assert str(lock) in captured.out


# ---------- Subprocess tests on the real writers ----------


@pytest.fixture
def live_lockfile():
    """Create the real lockfile for the duration of a test; always clean up."""
    LOCKFILE.touch()
    try:
        yield LOCKFILE
    finally:
        try:
            LOCKFILE.unlink()
        except FileNotFoundError:
            pass


def test_add_to_vectorstore_aborts_on_lockfile(live_lockfile):
    """Plan test #21: scripts/add_to_vectorstore.py aborts within 5s when locked."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "add_to_vectorstore.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}; stderr: {proc.stderr}"
    combined = proc.stdout + proc.stderr
    assert ABORT_MESSAGE_PREFIX in combined
    assert str(live_lockfile) in combined


def test_add_books_to_vectorstore_aborts_on_lockfile(live_lockfile):
    """Plan test #21: src/ingestion/process_books.py::add_books_to_vectorstore aborts when locked."""
    # Invoke the library function via python -c; the guard fires as the function's first statement.
    code = (
        "import sys; sys.path.insert(0, '"
        + str(REPO_ROOT)
        + "');\n"
        + "from src.ingestion.process_books import add_books_to_vectorstore;\n"
        + "add_books_to_vectorstore()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}; stderr: {proc.stderr}"
    combined = proc.stdout + proc.stderr
    assert ABORT_MESSAGE_PREFIX in combined
    assert str(live_lockfile) in combined


# ---------- Source-level assertion: guards are actually in place ----------


def test_writer_files_import_and_call_guard():
    """Defense in depth: grep the two writers for the guard symbol; if either removes it, this fails."""
    addvs = (REPO_ROOT / "scripts" / "add_to_vectorstore.py").read_text()
    procbk = (REPO_ROOT / "src" / "ingestion" / "process_books.py").read_text()
    for src in (addvs, procbk):
        assert "abort_if_migration_in_progress" in src, (
            "writer must import and call abort_if_migration_in_progress()"
        )
