"""SQLite metadata store.

A single guarded connection is shared between request handlers and the
background rebuild thread.  SQLite runs in WAL mode; an application-level
re-entrant lock additionally serialises multi-statement transactions so the
state machine (upload / overwrite / rebuild / disk transitions) cannot
interleave.  Tables cover files (with an overwrite version counter),
stripes/blocks, disks, rebuild jobs and crash-recovery records for patch
overwrite jobs.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    size         INTEGER NOT NULL,
    chunk_size   INTEGER NOT NULL,
    stripe_count INTEGER NOT NULL,
    sha256       TEXT,
    version      INTEGER NOT NULL DEFAULT 1,
    status       TEXT NOT NULL CHECK (status IN ('uploading', 'confirmed', 'overwriting')),
    created_at   REAL NOT NULL,
    confirmed_at REAL
);

CREATE TABLE IF NOT EXISTS stripes (
    file_id      TEXT NOT NULL,
    stripe_index INTEGER NOT NULL,
    length       INTEGER NOT NULL,
    d0_disk      INTEGER NOT NULL,
    d1_disk      INTEGER NOT NULL,
    p_disk       INTEGER NOT NULL,
    healthy      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (file_id, stripe_index),
    FOREIGN KEY (file_id) REFERENCES files(id)
);

CREATE TABLE IF NOT EXISTS blocks (
    file_id      TEXT NOT NULL,
    stripe_index INTEGER NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('d0', 'd1', 'p')),
    disk_index   INTEGER NOT NULL,
    relpath      TEXT NOT NULL,
    sha256       TEXT,
    PRIMARY KEY (file_id, stripe_index, role),
    FOREIGN KEY (file_id, stripe_index)
        REFERENCES stripes(file_id, stripe_index)
);

CREATE TABLE IF NOT EXISTS disks (
    disk_index  INTEGER PRIMARY KEY,
    state       TEXT NOT NULL CHECK (state IN ('online', 'offline', 'rebuilding')),
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rebuild_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    disk_index    INTEGER NOT NULL,
    state         TEXT NOT NULL CHECK (state IN ('running', 'done', 'failed')),
    stripes_total INTEGER NOT NULL DEFAULT 0,
    stripes_done  INTEGER NOT NULL DEFAULT 0,
    bytes_done    INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    started_at    REAL NOT NULL,
    finished_at   REAL
);

-- Crash-safe record of an in-progress (or finished) patch overwrite.
-- stage:
--   preparing  rollback data is being persisted; no live block touched yet
--   prepared   every old block backed up; live blocks may be rewritten
--   committed  all stripes rewritten; only the summary transaction remains
--   done       fully committed and visible as the new file version
CREATE TABLE IF NOT EXISTS overwrite_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       TEXT NOT NULL,
    state         TEXT NOT NULL CHECK (state IN ('running', 'done', 'failed')),
    stage         TEXT NOT NULL CHECK (stage IN
                    ('preparing', 'prepared', 'committed', 'done')),
    patch_offset  INTEGER NOT NULL,
    patch_size    INTEGER NOT NULL,
    rollback_dir  TEXT NOT NULL,
    error         TEXT,
    created_at    REAL NOT NULL,
    finished_at   REAL
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        existed = Path(self.path).exists()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=10000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            # executescript() commits any pending transaction first, which
            # matters for the PRAGMAs inside the legacy-schema migration.
            self.conn.executescript(SCHEMA)
            self._migrate_schema()
            now = time.time()
            with self.conn:
                for i in range(3):
                    self.conn.execute(
                        "INSERT OR IGNORE INTO disks(disk_index, state, "
                        "updated_at) VALUES (?, 'online', ?)",
                        (i, now),
                    )
        # True only when this process created the database file itself.
        self.is_fresh = not existed

    def _migrate_schema(self) -> None:
        """Upgrade databases created by older releases in place."""
        cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(files)")
        }
        if "version" not in cols:
            # Rebuild the table so the status CHECK admits 'overwriting' too.
            # The whole script runs via executescript(), which commits any
            # open transaction first: otherwise the foreign_keys PRAGMA would
            # be a silent no-op and DROP TABLE could fail on child rows.
            self.conn.executescript(
                "PRAGMA foreign_keys=OFF;\n"
                "CREATE TABLE files_new ("
                "id TEXT PRIMARY KEY, name TEXT NOT NULL, size INTEGER NOT NULL, "
                "chunk_size INTEGER NOT NULL, stripe_count INTEGER NOT NULL, "
                "sha256 TEXT, version INTEGER NOT NULL DEFAULT 1, "
                "status TEXT NOT NULL CHECK (status IN "
                "('uploading', 'confirmed', 'overwriting')), "
                "created_at REAL NOT NULL, confirmed_at REAL);\n"
                "INSERT INTO files_new(id, name, size, chunk_size, stripe_count, "
                "sha256, version, status, created_at, confirmed_at) "
                "SELECT id, name, size, chunk_size, stripe_count, sha256, 1, "
                "status, created_at, confirmed_at FROM files;\n"
                "DROP TABLE files;\n"
                "ALTER TABLE files_new RENAME TO files;\n"
                "PRAGMA foreign_keys=ON;\n"
            )

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # -- disks -----------------------------------------------------------
    def disk_states(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT disk_index, state, updated_at FROM disks ORDER BY disk_index"
            ).fetchall()
        return [dict(r) for r in rows]

    def disk_state(self, disk_index: int) -> str:
        with self._lock:
            row = self.conn.execute(
                "SELECT state FROM disks WHERE disk_index=?", (disk_index,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown disk {disk_index}")
        return row["state"]

    def set_disk_state(self, disk_index: int, state: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE disks SET state=?, updated_at=? WHERE disk_index=?",
                (state, time.time(), disk_index),
            )

    def offline_disk_indexes(self) -> list[int]:
        return [d["disk_index"] for d in self.disk_states() if d["state"] == "offline"]

    # -- files -----------------------------------------------------------
    def create_file(self, meta: dict[str, Any]) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO files(id, name, size, chunk_size, stripe_count, "
                "sha256, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    meta["id"],
                    meta["name"],
                    meta["size"],
                    meta["chunk_size"],
                    meta["stripe_count"],
                    meta.get("sha256"),
                    "uploading",
                    time.time(),
                ),
            )

    def confirm_file(self, file_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE files SET status='confirmed', confirmed_at=? WHERE id=?",
                (time.time(), file_id),
            )

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM files WHERE id=?", (file_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_files(self, only_confirmed: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM files"
        if only_confirmed:
            # An 'overwriting' file is still a previously confirmed file whose
            # download is blocked by the in-memory file lock, not by listing.
            sql += " WHERE status IN ('confirmed', 'overwriting')"
        sql += " ORDER BY created_at DESC"
        with self._lock:
            rows = self.conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def uploading_files(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM files WHERE status='uploading'"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_file_rows(self, file_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM overwrite_jobs WHERE file_id=?",
                              (file_id,))
            self.conn.execute("DELETE FROM blocks WHERE file_id=?", (file_id,))
            self.conn.execute("DELETE FROM stripes WHERE file_id=?", (file_id,))
            self.conn.execute("DELETE FROM files WHERE id=?", (file_id,))

    def set_file_status(self, file_id: str, status: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE files SET status=? WHERE id=?", (status, file_id)
            )

    def update_file_summary(self, file_id: str, sha256: str, version: int) -> None:
        """Commit the post-overwrite summary atomically (same txn as status)."""
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE files SET sha256=?, version=?, status='confirmed' "
                "WHERE id=?",
                (sha256, version, file_id),
            )

    def overwriting_files(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM files WHERE status='overwriting'"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- overwrite jobs --------------------------------------------------
    def create_overwrite_job(self, file_id: str, patch_offset: int,
                             patch_size: int, rollback_dir: str) -> int:
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO overwrite_jobs(file_id, state, stage, patch_offset, "
                "patch_size, rollback_dir, created_at) "
                "VALUES (?, 'running', 'preparing', ?, ?, ?, ?)",
                (file_id, patch_offset, patch_size, rollback_dir, time.time()),
            )
            return int(cur.lastrowid)

    def set_overwrite_stage(self, job_id: int, stage: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE overwrite_jobs SET stage=? WHERE id=?", (stage, job_id)
            )

    def finish_overwrite_job(self, job_id: int, state: str,
                             error: str | None) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE overwrite_jobs SET state=?, stage=CASE WHEN ?='done' "
                "THEN 'done' ELSE stage END, error=?, finished_at=? WHERE id=?",
                (state, state, error, time.time(), job_id),
            )

    def running_overwrite_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM overwrite_jobs WHERE state='running' ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def last_overwrite_job(self, file_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM overwrite_jobs WHERE file_id=? ORDER BY id DESC "
                "LIMIT 1",
                (file_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_overwrite_job(self, job_id: int) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM overwrite_jobs WHERE id=?", (job_id,))

    # -- stripes & blocks ------------------------------------------------
    def insert_stripe(self, stripe: dict[str, Any], healthy: bool) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO stripes(file_id, stripe_index, length, d0_disk, "
                "d1_disk, p_disk, healthy) VALUES (?,?,?,?,?,?,?)",
                (
                    stripe["file_id"],
                    stripe["stripe_index"],
                    stripe["length"],
                    stripe["d0_disk"],
                    stripe["d1_disk"],
                    stripe["p_disk"],
                    1 if healthy else 0,
                ),
            )

    def set_stripe_healthy(self, file_id: str, stripe_index: int, healthy: bool) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE stripes SET healthy=? WHERE file_id=? AND stripe_index=?",
                (1 if healthy else 0, file_id, stripe_index),
            )

    def get_stripe(self, file_id: str, stripe_index: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM stripes WHERE file_id=? AND stripe_index=?",
                (file_id, stripe_index),
            ).fetchone()
        return dict(row) if row else None

    def get_stripes(self, file_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM stripes WHERE file_id=? ORDER BY stripe_index",
                (file_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_block(self, block: dict[str, Any]) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO blocks(file_id, stripe_index, role, disk_index, "
                "relpath, sha256) VALUES (?,?,?,?,?,?)",
                (
                    block["file_id"],
                    block["stripe_index"],
                    block["role"],
                    block["disk_index"],
                    block["relpath"],
                    block.get("sha256"),
                ),
            )

    def get_blocks(self, file_id: str, stripe_index: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM blocks WHERE file_id=? AND stripe_index=?",
                (file_id, stripe_index),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_block(self, file_id: str, stripe_index: int, role: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM blocks WHERE file_id=? AND stripe_index=? AND role=?",
                (file_id, stripe_index, role),
            ).fetchone()
        return dict(row) if row else None

    def healthy_stripes_on_disk(self, disk_index: int) -> list[dict[str, Any]]:
        """All healthy stripes that place a block on ``disk_index``."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM stripes WHERE healthy=1 AND "
                "(d0_disk=? OR d1_disk=? OR p_disk=?) ORDER BY file_id, stripe_index",
                (disk_index, disk_index, disk_index),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_healthy_stripes_on_disk(self, disk_index: int) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS c FROM stripes WHERE healthy=1 AND "
                "(d0_disk=? OR d1_disk=? OR p_disk=?)",
                (disk_index, disk_index, disk_index),
            ).fetchone()
        return row["c"]

    # -- rebuild jobs ----------------------------------------------------
    def create_rebuild_job(self, disk_index: int, stripes_total: int) -> int:
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO rebuild_jobs(disk_index, state, stripes_total, "
                "stripes_done, bytes_done, started_at) VALUES (?, 'running', ?, 0, 0, ?)",
                (disk_index, stripes_total, time.time()),
            )
            return int(cur.lastrowid)

    def rebuild_progress(self, job_id: int, done: int, bytes_done: int,
                         total: int | None = None) -> None:
        with self._lock, self.conn:
            if total is None:
                self.conn.execute(
                    "UPDATE rebuild_jobs SET stripes_done=?, bytes_done=? WHERE id=?",
                    (done, bytes_done, job_id),
                )
            else:
                self.conn.execute(
                    "UPDATE rebuild_jobs SET stripes_done=?, bytes_done=?, "
                    "stripes_total=? WHERE id=?",
                    (done, bytes_done, total, job_id),
                )

    def finish_rebuild_job(self, job_id: int, state: str, error: str | None) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE rebuild_jobs SET state=?, error=?, finished_at=? WHERE id=?",
                (state, error, time.time(), job_id),
            )

    def running_rebuild_job(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM rebuild_jobs WHERE state='running' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def last_rebuild_job(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM rebuild_jobs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None
