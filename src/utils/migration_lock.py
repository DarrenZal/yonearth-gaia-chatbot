"""
Shared migration-lockfile guard — called by every Pinecone writer.

scripts/migrate_categories.py creates .pinecone-migration-lock at repo root
for the duration of a migration. Any writer that starts while the file exists
must abort immediately (exit 1) with a clear message. This prevents concurrent
upserts from polluting an in-flight metadata rewrite — the classic
"lost update" race.

Real writers that must call abort_if_migration_in_progress():
  - scripts/add_to_vectorstore.py (CLI entry for episodes)
  - src/ingestion/process_books.py::add_books_to_vectorstore (library)

Plan references: Rollback §, plan tests #21, AC #34.
"""
from __future__ import annotations

import sys
from pathlib import Path

# parents[2] from src/utils/<file>.py = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKFILE = REPO_ROOT / ".pinecone-migration-lock"

ABORT_MESSAGE_PREFIX = "ABORT: Pinecone migration in progress"


def abort_if_migration_in_progress(lockfile: Path | None = None) -> None:
    """Exit 1 with a clear message if the migration lockfile is present."""
    target = lockfile if lockfile is not None else LOCKFILE
    if target.exists():
        print(
            f"{ABORT_MESSAGE_PREFIX} (lockfile present at {target}). Refusing to write.",
            flush=True,
        )
        sys.exit(1)
