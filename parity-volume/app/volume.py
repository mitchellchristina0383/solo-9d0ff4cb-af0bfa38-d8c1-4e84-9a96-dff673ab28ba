"""Parity file volume core logic.

Implements the state machine on top of :mod:`db` and :mod:`storage`:

* streaming, crash-safe striped upload (2 data blocks + 1 XOR parity block)
* download with on-the-fly reconstruction from any two surviving blocks
* disk offlining, empty-disk replacement and background rebuild
* in-place patch overwrite of confirmed files (only touched stripes are
  rewritten; all-or-nothing via a rollback area)
* startup recovery: unconfirmed uploads are discarded, uncommitted
  overwrites are rolled back, interrupted rebuilds resume automatically
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from . import coding, storage
from .db import Database

logger = logging.getLogger("parity_volume")

# Test-only delay (seconds) applied after each rebuilt stripe so crash /
# restart scenarios can reliably interrupt an in-progress rebuild.
_REBUILD_STRIPE_DELAY = float(os.environ.get("PV_TEST_REBUILD_DELAY", "0"))

# Test-only fault injection: simulate a disk I/O error after the Nth stripe
# rewrite of an overwrite (0-based ordinal within the patch). Negative disables.
_OVERWRITE_IOFAIL_AFTER = int(
    os.environ.get("PV_TEST_OVERWRITE_IOFAIL_AFTER", "-1")
)

# Test-only gate: when an Event is assigned, an overwrite parks on it right
# after the first stripe rewrite, so a restart can be triggered while the
# overwrite is deterministically in flight (used for crash-recovery tests).
_OVERWRITE_PAUSE: threading.Event | None = None


class DiskUnavailableError(Exception):
    """Raised when an operation needs a disk that is offline/missing."""


class UnrecoverableError(Exception):
    """Raised when more than one disk is missing for needed data."""


class PatchOutOfBoundsError(Exception):
    """Raised when a patch would write past the end of its target file."""


class _OverwriteIOError(Exception):
    """Simulated disk write error for the online rollback self test."""


class _OverwriteWriteFailure(BaseException):
    """Injected failure used by the self tests to simulate a process kill.

    Inherits BaseException on purpose: it bypasses the normal rollback
    handler so the test can exercise startup recovery exactly like a real
    crash mid-overwrite.
    """


class Volume:
    def __init__(self, base_dir: str | Path, db_path: str | Path,
                 chunk_size: int = 512 * 1024):
        self.base_dir = Path(base_dir)
        self.chunk_size = chunk_size
        self.db = Database(db_path)
        self.roots = [storage.disk_root(self.base_dir, i) for i in range(coding.DISK_COUNT)]
        self._rebuild_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Per-file locks serialise patch overwrite against download/delete so
        # a download always observes either the complete old or new content.
        self._file_locks: dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()
        # Rollback backups live next to the metadata DB, which is on a fourth
        # independent directory and unaffected by any single data disk loss.
        self.rollback_root = Path(self.db.path).parent / "overwrite-rollback"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        # A freshly created database gets the default layout.  For an existing
        # deployment, detect vanished disk directories first so a real disk
        # loss is not masked by silently re-creating the directory.
        if not self.db.is_fresh:
            self._mark_missing_disks_offline()
        for i, root in enumerate(self.roots):
            if self.db.disk_state(i) == "online":
                storage.ensure_disk_layout(root)
        self._purge_unconfirmed_uploads()
        self._recover_interrupted_overwrites()
        job = self.db.running_rebuild_job()
        if job is not None:
            self._spawn_rebuild(job["disk_index"], resume=True)

    def _reconcile_disk_presence(self) -> None:
        """Runtime check for disk directories appearing or vanishing.

        * An online disk whose directory was deleted becomes ``offline``.
        * A non-online disk whose directory exists (operator mounted a
          replacement) gets its ``blocks`` layout ensured.
        """
        for i, root in enumerate(self.roots):
            state = self.db.disk_state(i)
            if not root.is_dir():
                if state == "online":
                    logger.warning("disk%d directory vanished, marking offline", i)
                    self.db.set_disk_state(i, "offline")
            elif state != "online":
                storage.ensure_disk_layout(root)

    def shutdown(self) -> None:
        self._stop_event.set()
        thread = self._rebuild_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=30)
        self.db.close()

    def _mark_missing_disks_offline(self) -> None:
        """A disk directory that vanished (e.g. container restart without the
        volume mounted) is treated as an offline disk."""
        for i, root in enumerate(self.roots):
            if not root.is_dir():
                if self.db.disk_state(i) != "offline":
                    logger.warning("disk%d directory missing, marking offline", i)
                    self.db.set_disk_state(i, "offline")

    # ------------------------------------------------------------------ #
    # Disk operations
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        self._reconcile_disk_presence()
        disks = self.db.disk_states()
        job = self.db.running_rebuild_job()
        for d in disks:
            d["path"] = str(self.roots[d["disk_index"]])
            d["present"] = self.roots[d["disk_index"]].is_dir()
        return {
            "disks": disks,
            "rebuild": job,
            "chunk_size": self.chunk_size,
            "files": len(self.db.list_files()),
        }

    def take_disk_offline(self, disk_index: int) -> None:
        self._check_disk_index(disk_index)
        state = self.db.disk_state(disk_index)
        if state == "offline":
            return
        if state == "rebuilding":
            raise DiskUnavailableError(
                f"disk{disk_index} is being rebuilt and cannot be taken offline"
            )
        if self.db.running_rebuild_job() is not None:
            raise DiskUnavailableError("a rebuild is already in progress")
        self.db.set_disk_state(disk_index, "offline")
        logger.info("disk%d taken offline", disk_index)

    def replace_disk(self, disk_index: int) -> None:
        """Replace an offline disk with an empty one and start rebuilding."""
        self._check_disk_index(disk_index)
        state = self.db.disk_state(disk_index)
        if state != "offline":
            raise DiskUnavailableError(
                f"disk{disk_index} must be offline before replacement (state={state})"
            )
        if self.db.running_rebuild_job() is not None:
            raise DiskUnavailableError("a rebuild is already in progress")
        # Archive the old directory and provision an empty replacement.
        storage.replace_with_empty_disk(self.roots[disk_index])
        total = self.db.count_healthy_stripes_on_disk(disk_index)
        job_id = self.db.create_rebuild_job(disk_index, total)
        self.db.set_disk_state(disk_index, "rebuilding")
        logger.info("disk%d replaced, rebuild job %d over %d stripes",
                    disk_index, job_id, total)
        self._spawn_rebuild(disk_index)

    def _spawn_rebuild(self, disk_index: int, resume: bool = False) -> None:
        if self.db.disk_state(disk_index) != "rebuilding":
            self.db.set_disk_state(disk_index, "rebuilding")
        self._stop_event = threading.Event()
        self._rebuild_thread = threading.Thread(
            target=self._rebuild_worker,
            args=(disk_index,),
            name=f"rebuild-disk{disk_index}",
            daemon=True,
        )
        self._rebuild_thread.start()
        if resume:
            logger.info("resumed rebuild of disk%d", disk_index)

    # ------------------------------------------------------------------ #
    # Upload
    # ------------------------------------------------------------------ #
    def upload_file(self, fileobj: BinaryIO, filename: str) -> dict[str, Any]:
        file_id = uuid.uuid4().hex
        size = 0
        digest = hashlib.sha256()
        stripe_index = 0
        created = False
        try:
            self.db.create_file({
                "id": file_id,
                "name": filename,
                "size": 0,
                "chunk_size": self.chunk_size,
                "stripe_count": 0,
            })
            created = True
            while True:
                chunk = self._read_exact(fileobj, self.chunk_size, digest)
                if not chunk:
                    break
                size += len(chunk)
                self._write_stripe(file_id, stripe_index, chunk)
                stripe_index += 1
            stripe_count = stripe_index
            # Fix file size / stripe count, then make the file visible.
            with self.db.lock, self.db.conn:
                self.db.conn.execute(
                    "UPDATE files SET size=?, stripe_count=?, sha256=? WHERE id=?",
                    (size, stripe_count, digest.hexdigest(), file_id),
                )
            self.db.confirm_file(file_id)
            logger.info("uploaded %s (%d bytes, %d stripes)",
                        filename, size, stripe_count)
            return self.db.get_file(file_id)  # type: ignore[return-value]
        except Exception:
            if created:
                self._discard_file(file_id)
            raise

    @staticmethod
    def _read_exact(fileobj: BinaryIO, size: int,
                    digest: hashlib._Hash) -> bytes:
        """Read exactly ``size`` bytes unless EOF is reached."""
        parts: list[bytes] = []
        remaining = size
        while remaining:
            part = fileobj.read(remaining)
            if not part:
                break
            digest.update(part)
            parts.append(part)
            remaining -= len(part)
        return b"".join(parts)

    def _stripe_layout(self, stripe_index: int) -> tuple[int, int, int]:
        """Rotate which disk holds the parity block across stripes."""
        p_disk = stripe_index % coding.DISK_COUNT
        data_disks = [d for d in range(coding.DISK_COUNT) if d != p_disk]
        d0_disk, d1_disk = data_disks
        return d0_disk, d1_disk, p_disk

    def _write_stripe(self, file_id: str, stripe_index: int, data: bytes) -> None:
        length = len(data)
        # Zero-pad so all three blocks have identical length.
        d0_data = data + b"\x00" * (self.chunk_size - length)
        d1_data = b"\x00" * self.chunk_size
        d0, d1, p = self._stripe_layout(stripe_index)
        layout = {"d0": (d0, d0_data), "d1": (d1, d1_data),
                  "p": (p, coding.parity(d0_data, d1_data))}
        # A stripe is only marked healthy once all three blocks are durable,
        # so a crash can never leave a partial stripe flagged as healthy.
        self._reconcile_disk_presence()
        self.db.insert_stripe({
            "file_id": file_id,
            "stripe_index": stripe_index,
            "length": length,
            "d0_disk": d0,
            "d1_disk": d1,
            "p_disk": p,
        }, healthy=False)
        for role, (disk_index, payload) in layout.items():
            # An 'offline' disk has no usable directory; a 'rebuilding' disk
            # holds the empty replacement and accepts new stripes normally.
            if self.db.disk_state(disk_index) == "offline":
                raise DiskUnavailableError(
                    f"disk{disk_index} is offline; refusing partial stripe"
                )
            relpath = storage.block_relpath(file_id, stripe_index, role)
            storage.write_block(self.roots[disk_index], relpath, payload)
            self.db.insert_block({
                "file_id": file_id,
                "stripe_index": stripe_index,
                "role": role,
                "disk_index": disk_index,
                "relpath": relpath,
            })
        self.db.set_stripe_healthy(file_id, stripe_index, True)

    def _discard_file(self, file_id: str) -> None:
        """Remove every block and metadata row of an unconfirmed file."""
        for stripe in self.db.get_stripes(file_id):
            for block in self.db.get_blocks(file_id, stripe["stripe_index"]):
                storage.remove_block(self.roots[block["disk_index"]],
                                     block["relpath"])
        self.db.delete_file_rows(file_id)

    def _purge_unconfirmed_uploads(self) -> None:
        """Crash recovery: nothing unconfirmed may ever become visible."""
        for meta in self.db.uploading_files():
            logger.info("purging unconfirmed upload %s", meta["id"])
            self._discard_file(meta["id"])

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #
    def list_files(self) -> list[dict[str, Any]]:
        return self.db.list_files()

    def delete_file(self, file_id: str) -> bool:
        """Delete a file and all of its blocks. Returns False if unknown."""
        if self.db.get_file(file_id) is None:
            return False
        with self._file_lock(file_id):
            if self.db.get_file(file_id) is None:
                return False
            self._discard_file(file_id)
        return True

    # ------------------------------------------------------------------ #
    # Patch overwrite
    # ------------------------------------------------------------------ #
    def _file_lock(self, file_id: str) -> threading.Lock:
        with self._file_locks_guard:
            lock = self._file_locks.get(file_id)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[file_id] = lock
            return lock

    def _require_all_disks_online(self) -> None:
        self._reconcile_disk_presence()
        not_online = [
            i for i in range(coding.DISK_COUNT)
            if self.db.disk_state(i) != "online"
        ]
        if not_online:
            raise DiskUnavailableError(
                f"overwrite requires all three disks online; "
                f"disk(s) {not_online} are not online"
            )

    def apply_patch(self, file_id: str, offset: int,
                    patch: bytes) -> dict[str, Any]:
        """Overwrite ``[offset, offset+len(patch))`` of a confirmed file.

        Only the stripes touched by the patch are rewritten; the file id and
        its length are preserved. The whole operation is all-or-nothing: on
        write failure or service restart the old content is restored before
        the file becomes readable again.
        """
        meta = self.db.get_file(file_id)
        if meta is None or meta["status"] not in ("confirmed", "overwriting"):
            raise FileNotFoundError(file_id)
        if offset < 0:
            raise PatchOutOfBoundsError("offset must be >= 0")
        if not patch:
            raise PatchOutOfBoundsError("patch must not be empty")

        with self._file_lock(file_id):
            # Re-read inside the lock so queued patches observe the version
            # and size committed by the patch that ran ahead of them.
            meta = self.db.get_file(file_id)
            if meta is None or meta["status"] not in (
                "confirmed", "overwriting"
            ):
                raise FileNotFoundError(file_id)
            end = offset + len(patch)
            if end > meta["size"]:
                raise PatchOutOfBoundsError(
                    f"patch range [{offset}, {end}) exceeds file size "
                    f"{meta['size']}"
                )
            self._require_all_disks_online()
            if self.db.running_overwrite_jobs():
                raise UnrecoverableError(
                    "a previous overwrite did not finish rollback; "
                    "restart the service to complete recovery"
                )
            rollback_dir = self.rollback_root / (
                f"job-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:10]}"
            )
            rollback_dir.mkdir(parents=True, exist_ok=False)
            job_id = self.db.create_overwrite_job(
                file_id, offset, len(patch), str(rollback_dir)
            )
            self.db.set_file_status(file_id, "overwriting")
            try:
                first = offset // self.chunk_size
                last = (end - 1) // self.chunk_size
                affected = list(range(first, last + 1))

                # Phase 1: persist a backup of every block we may rewrite.
                # Until this completes no live block is touched.
                for idx in affected:
                    self._backup_stripe(file_id, idx, rollback_dir)
                self.db.set_overwrite_stage(job_id, "prepared")

                # Phase 2: rewrite only the stripes touched by the patch.
                ordinal = 0
                for idx in affected:
                    self._rewrite_stripe_with_patch(
                        file_id, idx, offset, patch
                    )
                    if (
                        _OVERWRITE_IOFAIL_AFTER >= 0
                        and ordinal == _OVERWRITE_IOFAIL_AFTER
                    ):
                        raise _OverwriteIOError(
                            f"simulated disk write error on stripe {idx}"
                        )
                    if _OVERWRITE_PAUSE is not None and ordinal == 0:
                        # Deterministic crash window for the restart test.
                        _OVERWRITE_PAUSE.wait(timeout=30)
                        raise _OverwriteWriteFailure(
                            "simulated process kill mid-overwrite"
                        )
                    ordinal += 1
                self.db.set_overwrite_stage(job_id, "committed")

                # Phase 3: recompute the summary and atomically publish.
                new_digest = self._recompute_sha256(file_id)
                new_version = int(meta["version"]) + 1
                with self.db.lock, self.db.conn:
                    self.db.update_file_summary(file_id, new_digest, new_version)
                    self.db.finish_overwrite_job(job_id, "done", None)
                shutil.rmtree(rollback_dir, ignore_errors=True)
                logger.info(
                    "patched %s at offset %d (%d bytes, %d stripes, v%d)",
                    file_id, offset, len(patch), len(affected), new_version,
                )
                return self.db.get_file(file_id)  # type: ignore[return-value]
            except Exception as exc:
                logger.error("overwrite job %d failed: %s; rolling back",
                             job_id, exc)
                rollback_error = self._rollback_overwrite(
                    self.db.last_overwrite_job(file_id)
                )
                self.db.set_file_status(file_id, "confirmed")
                self.db.finish_overwrite_job(
                    job_id, "failed",
                    str(exc) if rollback_error is None
                    else f"{exc}; rollback error: {rollback_error}",
                )
                shutil.rmtree(rollback_dir, ignore_errors=True)
                if rollback_error is not None:
                    raise UnrecoverableError(
                        f"overwrite failed and rollback could not complete: "
                        f"{rollback_error}"
                    ) from exc
                raise

    def _backup_stripe(self, file_id: str, stripe_index: int,
                       rollback_dir: Path) -> None:
        """Copy all three live blocks of a stripe into the rollback area."""
        stripe = self.db.get_stripe(file_id, stripe_index)
        if stripe is None or not stripe["healthy"]:
            raise UnrecoverableError(
                f"stripe {file_id}/{stripe_index} missing or unhealthy"
            )
        out_dir = rollback_dir / f"stripe-{stripe_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for role in coding.ROLES:
            block = self.db.get_block(file_id, stripe_index, role)
            if block is None:
                raise UnrecoverableError(
                    f"missing {role} block for {file_id}/{stripe_index}"
                )
            src = self.roots[block["disk_index"]] / block["relpath"]
            data = storage.read_block(self.roots[block["disk_index"]],
                                      block["relpath"])
            storage.write_block(out_dir, f"{role}.block", data)
            if not src.is_file():
                raise UnrecoverableError(
                    f"block vanished during backup: {src}"
                )

    def _rewrite_stripe_with_patch(self, file_id: str, stripe_index: int,
                                   offset: int, patch: bytes) -> None:
        stripe = self.db.get_stripe(file_id, stripe_index)
        if stripe is None:
            raise UnrecoverableError(
                f"stripe {file_id}/{stripe_index} vanished during overwrite"
            )
        # Read the current stripe data; tolerates a concurrently offline disk
        # only insofar as two blocks remain -- but callers already required
        # all three disks online, so this is the normal fast path.
        old_d0 = self._read_stripe_data(file_id, stripe)
        buf = bytearray(old_d0)
        stripe_start = stripe_index * self.chunk_size
        local_lo = max(offset, stripe_start) - stripe_start
        local_hi = min(offset + len(patch), stripe_start + self.chunk_size) \
            - stripe_start
        patch_lo = max(stripe_start, offset) - offset
        buf[local_lo:local_hi] = patch[
            patch_lo:patch_lo + (local_hi - local_lo)
        ]
        new_d0 = bytes(buf)
        # d1 is always a zero block for every stripe, so only d0 and the
        # parity block change.
        new_p = coding.parity(new_d0, b"\x00" * self.chunk_size)

        role_writes = (
            ("d0", stripe["d0_disk"], new_d0),
            ("p", stripe["p_disk"], new_p),
        )
        for role, disk_index, payload in role_writes:
            # Re-check right before each physical write: a disk that
            # disappeared mid-overwrite must not leave a half-rewritten set
            # without triggering rollback.
            if self.db.disk_state(disk_index) != "online":
                raise DiskUnavailableError(
                    f"disk{disk_index} not online during overwrite"
                )
            block = self.db.get_block(file_id, stripe_index, role)
            if block is None:
                raise UnrecoverableError(
                    f"{role} block metadata missing for "
                    f"{file_id}/{stripe_index}"
                )
            storage.write_block(
                self.roots[disk_index], block["relpath"], payload
            )

    def _rollback_overwrite(self, job: dict[str, Any] | None) -> str | None:
        """Restore every backed-up block. Returns an error string on failure."""
        if job is None:
            return "overwrite job metadata missing"
        rollback_dir = Path(job["rollback_dir"])
        if not rollback_dir.is_dir():
            # 'preparing' stage: no live block had been rewritten yet.
            if job["stage"] == "preparing":
                return None
            return f"rollback directory {rollback_dir} is missing"
        file_id = job["file_id"]
        for stripe_dir in sorted(rollback_dir.glob("stripe-*")):
            try:
                stripe_index = int(stripe_dir.name.split("-", 1)[1])
            except (IndexError, ValueError):
                return f"malformed rollback entry {stripe_dir.name}"
            for role in coding.ROLES:
                backup = stripe_dir / f"{role}.block"
                if not backup.is_file():
                    return f"backup missing: {backup}"
                block = self.db.get_block(file_id, stripe_index, role)
                if block is None:
                    return (f"block metadata missing for "
                            f"{file_id}/{stripe_index}/{role}")
                try:
                    storage.write_block(
                        self.roots[block["disk_index"]],
                        block["relpath"],
                        storage.read_block(stripe_dir, f"{role}.block"),
                    )
                except OSError as exc:
                    return f"cannot restore {file_id}/{stripe_index}/{role}: {exc}"
        return None

    def _recover_interrupted_overwrites(self) -> None:
        """Crash recovery: every uncommitted overwrite is fully undone."""
        for job in self.db.running_overwrite_jobs():
            file_id = job["file_id"]
            logger.warning(
                "recovering interrupted overwrite job %d on %s (stage=%s)",
                job["id"], file_id, job["stage"],
            )
            err = self._rollback_overwrite(job)
            if err is not None:
                # Keep the job running so a later restart retries the
                # rollback; leave the file hidden from downloads.
                logger.critical(
                    "rollback for overwrite job %d incomplete: %s",
                    job["id"], err,
                )
                self.db.finish_overwrite_job(job["id"], "failed", err)
                continue
            self.db.set_file_status(file_id, "confirmed")
            self.db.finish_overwrite_job(job["id"], "failed",
                                         "interrupted; rolled back at startup")
            shutil.rmtree(job["rollback_dir"], ignore_errors=True)
            logger.info("overwrite job %d rolled back at startup", job["id"])

    def _recompute_sha256(self, file_id: str) -> str:
        """Stream the whole (patched) file through sha256."""
        digest = hashlib.sha256()
        for stripe in self.db.get_stripes(file_id):
            data = self._read_stripe_data(file_id, stripe)
            digest.update(data[:stripe["length"]])
        return digest.hexdigest()

    def open_file_stripes(self, file_id: str) -> tuple[dict[str, Any], BinaryIO]:
        meta = self.db.get_file(file_id)
        if meta is None or meta["status"] != "confirmed":
            raise FileNotFoundError(file_id)
        # Serialise against in-progress patch overwrites: the lock is held for
        # the whole rewrite, so a download either precedes it (all old blocks)
        # or follows it (all new blocks), never a mixture.
        with self._file_lock(file_id):
            meta = self.db.get_file(file_id)
            if meta is None or meta["status"] != "confirmed":
                raise FileNotFoundError(file_id)
            stripes = self.db.get_stripes(file_id)
            if len(stripes) != meta["stripe_count"]:
                raise UnrecoverableError(
                    f"metadata incomplete for {file_id}: "
                    f"{len(stripes)}/{meta['stripe_count']} stripes"
                )
            from tempfile import SpooledTemporaryFile

            out = SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
            written = 0
            for stripe in stripes:
                data = self._read_stripe_data(file_id, stripe)
                out.write(data[:stripe["length"]])
                written += stripe["length"]
            if written != meta["size"]:
                out.close()
                raise UnrecoverableError(
                    f"reconstructed size {written} != metadata size {meta['size']}"
                )
            out.seek(0)
            return meta, out

    def _read_stripe_data(self, file_id: str, stripe: dict[str, Any]) -> bytes:
        disk_of_role = {
            "d0": stripe["d0_disk"],
            "d1": stripe["d1_disk"],
            "p": stripe["p_disk"],
        }
        available: dict[str, bytes] = {}
        missing_roles: list[str] = []
        for role, disk_index in disk_of_role.items():
            block = self.db.get_block(file_id, stripe["stripe_index"], role)
            if block is None:
                missing_roles.append(role)
                continue
            state = self.db.disk_state(disk_index)
            if state == "offline" or not storage.block_exists(
                self.roots[disk_index], block["relpath"]
            ):
                missing_roles.append(role)
                continue
            available[role] = storage.read_block(self.roots[disk_index],
                                                 block["relpath"])
        if len(available) < 2:
            raise UnrecoverableError(
                f"stripe {file_id}/{stripe['stripe_index']} has only "
                f"{len(available)} readable block(s); more than one disk failed"
            )
        if len(available) == 3:
            check = coding.xor_bytes(
                coding.xor_bytes(available["d0"], available["d1"]),
                available["p"],
            )
            if any(check):
                raise UnrecoverableError(
                    f"stripe {file_id}/{stripe['stripe_index']} failed parity check"
                )
            return available["d0"]
        # Exactly two blocks: reconstruct the missing one.  If a data block is
        # missing we return the reconstructed data; otherwise use d0 directly.
        missing_role = missing_roles[0]
        rebuilt = coding.rebuild_missing(missing_role, available)
        if missing_role == "d0":
            return rebuilt
        if missing_role == "d1":
            return available["d0"]
        return available["d0"]

    # ------------------------------------------------------------------ #
    # Rebuild
    # ------------------------------------------------------------------ #
    def _rebuild_worker(self, disk_index: int) -> None:
        job = self.db.running_rebuild_job()
        if job is None or job["disk_index"] != disk_index:
            logger.error("no running rebuild job for disk%d", disk_index)
            return
        job_id = job["id"]
        done = 0
        bytes_done = 0
        try:
            for stripe in self.db.healthy_stripes_on_disk(disk_index):
                if self._stop_event.is_set():
                    logger.info("rebuild of disk%d interrupted", disk_index)
                    return
                self._rebuild_stripe(disk_index, stripe)
                if _REBUILD_STRIPE_DELAY:
                    time.sleep(_REBUILD_STRIPE_DELAY)
                done += 1
                bytes_done += self.chunk_size
                if done % 8 == 0:
                    self.db.rebuild_progress(job_id, done, bytes_done)
            self.db.rebuild_progress(job_id, done, bytes_done)
            # Re-scan: stripes created by uploads interleaved with the rebuild
            # must all have a block on the new disk before we declare success.
            remaining = self._stripes_missing_on_disk(disk_index)
            if remaining:
                raise UnrecoverableError(
                    f"{len(remaining)} stripes still missing on disk{disk_index}"
                )
            # Interleaved uploads may have added stripes beyond the original
            # total; sync the counters so progress ends at exactly 100%.
            final_total = self.db.count_healthy_stripes_on_disk(disk_index)
            self.db.rebuild_progress(job_id, final_total,
                                     bytes_done, total=final_total)
            self.db.finish_rebuild_job(job_id, "done", None)
            self.db.set_disk_state(disk_index, "online")
            logger.info("rebuild of disk%d complete (%d stripes)",
                        disk_index, done)
        except Exception as exc:  # noqa: BLE001 - recorded on the job
            logger.exception("rebuild of disk%d failed", disk_index)
            self.db.finish_rebuild_job(job_id, "failed", str(exc))
            self.db.set_disk_state(disk_index, "offline")

    def _rebuild_stripe(self, disk_index: int, stripe: dict[str, Any]) -> None:
        file_id = stripe["file_id"]
        idx = stripe["stripe_index"]
        disk_of_role = {
            "d0": stripe["d0_disk"],
            "d1": stripe["d1_disk"],
            "p": stripe["p_disk"],
        }
        role_on_disk = next(
            role for role, disk in disk_of_role.items() if disk == disk_index
        )
        survivors: dict[str, bytes] = {}
        for role, other_disk in disk_of_role.items():
            if role == role_on_disk:
                continue
            block = self.db.get_block(file_id, idx, role)
            if block is None or self.db.disk_state(other_disk) != "online":
                raise UnrecoverableError(
                    f"cannot rebuild stripe {file_id}/{idx}: disk{other_disk} "
                    f"for role {role} is unavailable"
                )
            if not storage.block_exists(self.roots[other_disk], block["relpath"]):
                raise UnrecoverableError(
                    f"block file missing on disk{other_disk} for "
                    f"stripe {file_id}/{idx}"
                )
            survivors[role] = storage.read_block(self.roots[other_disk],
                                                 block["relpath"])
        rebuilt = coding.rebuild_missing(role_on_disk, survivors)
        relpath = storage.block_relpath(file_id, idx, role_on_disk)
        # The replacement disk was marked rebuilding; it is writable here.
        storage.write_block(self.roots[disk_index], relpath, rebuilt)
        block = self.db.get_block(file_id, idx, role_on_disk)
        if block is None:
            self.db.insert_block({
                "file_id": file_id,
                "stripe_index": idx,
                "role": role_on_disk,
                "disk_index": disk_index,
                "relpath": relpath,
            })

    def _stripes_missing_on_disk(self, disk_index: int) -> list[dict[str, Any]]:
        missing: list[dict[str, Any]] = []
        for stripe in self.db.healthy_stripes_on_disk(disk_index):
            role_on_disk = next(
                role for role, disk in (
                    ("d0", stripe["d0_disk"]),
                    ("d1", stripe["d1_disk"]),
                    ("p", stripe["p_disk"]),
                ) if disk == disk_index
            )
            block = self.db.get_block(stripe["file_id"],
                                      stripe["stripe_index"], role_on_disk)
            if block is None or not storage.block_exists(
                self.roots[disk_index], block["relpath"]
            ):
                missing.append(stripe)
        return missing

    # ------------------------------------------------------------------ #
    def _check_disk_index(self, disk_index: int) -> None:
        if disk_index not in range(coding.DISK_COUNT):
            raise ValueError(f"disk index must be 0..{coding.DISK_COUNT - 1}")
