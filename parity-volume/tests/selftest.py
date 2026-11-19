"""End-to-end self tests for the parity file volume.

Run with:  PV_EXTRA_SYS_PATH=/tmp/pylibs python -I tests/selftest.py

A real uvicorn server is started in a background thread (same process, so the
Volume object is reachable in-process for fault injection); all traffic goes
over real HTTP against temporary on-disk directories.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_extra = os.environ.get("PV_EXTRA_SYS_PATH")
if _extra:
    for p in reversed(_extra.split(os.pathsep)):
        sys.path.insert(0, p)

_TMP = Path(tempfile.mkdtemp(prefix="pv-selftest-"))
os.environ["PV_DATA_DIR"] = str(_TMP / "data")
os.environ["PV_DB_PATH"] = str(_TMP / "data" / "meta" / "volume.db")
os.environ["PV_CHUNK_SIZE"] = str(64 * 1024)  # 64 KiB: many stripes quickly

import httpx  # noqa: E402
import uvicorn  # noqa: E402

import app.main as webapp  # noqa: E402
import app.volume as volmod  # noqa: E402
from app import coding, storage  # noqa: E402
from app.volume import Volume  # noqa: E402

PASS = 0
FAIL = 0
PORT = int(os.environ.get("PV_TEST_PORT", "8765"))
BASE = f"http://127.0.0.1:{PORT}"
DATA_DIR = _TMP / "data"
DB_PATH = _TMP / "data" / "meta" / "volume.db"
CHUNK = 64 * 1024


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=30)


def vol() -> Volume:
    return webapp.volume


def upload(cli: httpx.Client, data: bytes, name: str) -> str:
    r = cli.post("/api/files",
                 files={"file": (name, io.BytesIO(data), "application/octet-stream")})
    r.raise_for_status()
    return r.json()["file"]["id"]


def download(cli: httpx.Client, file_id: str) -> bytes:
    r = cli.get(f"/api/files/{file_id}")
    if r.status_code != 200:
        raise AssertionError(f"download {file_id} -> {r.status_code} {r.text}")
    return r.content


def patch(cli: httpx.Client, file_id: str, offset: int,
          data: bytes) -> httpx.Response:
    return cli.post(
        f"/api/files/{file_id}/patch",
        data={"offset": str(offset)},
        files={"patch": ("p.bin", io.BytesIO(data),
                         "application/octet-stream")},
    )


def patched(original: bytes, offset: int, payload: bytes) -> bytes:
    out = bytearray(original)
    out[offset:offset + len(payload)] = payload
    return bytes(out)


def all_block_bytes(v: Volume) -> dict[tuple[int, str], bytes]:
    """Snapshot every physical block, keyed by (disk, relative path)."""
    snap: dict[tuple[int, str], bytes] = {}
    for disk_index, root in enumerate(v.roots):
        for path in root.rglob("*.block"):
            snap[(disk_index, str(path.relative_to(root)))] = path.read_bytes()
    return snap


def wait_rebuild(timeout: float = 60.0) -> dict:
    """Wait for the most recently created rebuild job to reach a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = vol().db.last_rebuild_job()
        if job is not None and job["state"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("rebuild did not finish")


def restart_service() -> None:
    """Simulate a full service restart in-process."""
    old = webapp.volume
    old.shutdown()
    webapp.volume = Volume(DATA_DIR, DB_PATH, chunk_size=CHUNK)
    webapp.volume.start()


def main() -> int:
    config = uvicorn.Config("app.main:app", host="127.0.0.1", port=PORT,
                            log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for the HTTP listener.
    for _ in range(100):
        try:
            httpx.get(f"{BASE}/api/status", timeout=1)
            break
        except httpx.TransportError:
            time.sleep(0.1)

    try:
        with client() as cli:
            # ---- startup ----------------------------------------------
            section("启动与空状态")
            st = cli.get("/api/status").json()
            check("三块盘初始在线",
                  [d["state"] for d in st["disks"]] == ["online"] * 3)
            check("初始无文件", cli.get("/api/files").json()["files"] == [])

            # ---- uploads ----------------------------------------------
            section("上传与完整性")
            payload_a = os.urandom(300_000)      # 5 stripes, last partial
            payload_b = os.urandom(CHUNK * 3 + 7)
            id_a = upload(cli, payload_a, "alpha.bin")
            id_b = upload(cli, payload_b, "beta.bin")
            id_e = upload(cli, b"", "empty.bin")
            check("下载 alpha 内容一致", download(cli, id_a) == payload_a)
            check("下载 beta 内容一致", download(cli, id_b) == payload_b)
            check("空文件可下载", download(cli, id_e) == b"")
            check("文件列表只含已确认文件",
                  {f["name"] for f in cli.get("/api/files").json()["files"]}
                  == {"alpha.bin", "beta.bin", "empty.bin"})
            check("sha256 已记录",
                  all(f["sha256"] for f in cli.get("/api/files").json()["files"]))

            block_dirs = list(_TMP.glob("data/disk*/blocks"))
            counts = [sum(1 for _ in d.rglob("*.block")) for d in block_dirs]
            check("三块盘上都有块文件", all(c > 0 for c in counts), str(counts))

            # ---- single disk failure for each disk --------------------
            section("单盘下线后读取（逐盘演练）")
            for dead in range(3):
                r = cli.post(f"/api/disks/{dead}/offline")
                check(f"disk{dead} 下线成功", r.status_code == 200)
                check(f"disk{dead} 下线后 alpha 可恢复",
                      download(cli, id_a) == payload_a)
                check(f"disk{dead} 下线后 beta 可恢复",
                      download(cli, id_b) == payload_b)

                # ---- two-disk failure is explicitly unrecoverable -----
                other = (dead + 1) % 3
                vol().db.set_disk_state(other, "offline")  # fault injection
                r = cli.get(f"/api/files/{id_a}")
                check("两盘失效明确报不可恢复(503 unrecoverable)",
                      r.status_code == 503
                      and r.json()["error"] == "unrecoverable", r.text)
                vol().db.set_disk_state(other, "online")

                # ---- replace + rebuild --------------------------------
                section(f"disk{dead} 换空盘与重建")
                r = cli.post(f"/api/disks/{dead}/replace")
                check(f"disk{dead} 换盘并启动重建", r.status_code == 200, r.text)
                job = wait_rebuild()
                check("重建任务成功", job.get("state") == "done", str(job))
                check("重建后磁盘回到 online",
                      vol().db.disk_state(dead) == "online")
                check("重建后 alpha 内容一致", download(cli, id_a) == payload_a)
                check("重建后 beta 内容一致", download(cli, id_b) == payload_b)

                all_three = True
                for f in (id_a, id_b):
                    for s in vol().db.get_stripes(f):
                        for b in vol().db.get_blocks(f, s["stripe_index"]):
                            if not storage.block_exists(
                                    vol().roots[b["disk_index"]], b["relpath"]):
                                all_three = False
                check("重建后每条条纹三盘齐备", all_three)

            # ---- rebuild interleaved with upload ----------------------
            section("重建与上传交错：新条带不遗漏")
            cli.post("/api/disks/1/offline").raise_for_status()
            r = cli.post("/api/disks/1/replace")
            check("重建已启动", r.status_code == 200, r.text)
            new_ids, new_payloads = [], {}
            for k in range(4):
                p = os.urandom(150_000 + k * 1234)
                fid = upload(cli, p, f"concurrent-{k}.bin")
                new_ids.append(fid)
                new_payloads[fid] = p
                check(f"并发上传 {k} 立即可下载", download(cli, fid) == p)
            job = wait_rebuild()
            check("交错场景重建成功", job.get("state") == "done", str(job))
            check("重建收尾扫描确认无缺失条带",
                  vol()._stripes_missing_on_disk(1) == [])
            for fid in new_ids:
                check(f"新文件 {fid[:8]} 重建后可读且容错",
                      download(cli, fid) == new_payloads[fid])

            cli.post("/api/disks/2/offline").raise_for_status()
            check("再坏一盘，新文件仍可下载",
                  all(download(cli, fid) == new_payloads[fid] for fid in new_ids))
            check("旧文件在再坏一盘时仍可下载",
                  download(cli, id_a) == payload_a
                  and download(cli, id_b) == payload_b)
            cli.post("/api/disks/2/replace").raise_for_status()
            check("第二次重建成功", wait_rebuild().get("state") == "done")

            # ---- patch overwrite ----------------------------------------
            section("补丁覆盖：单条带 / 边界 / 跨条带 / 只重写涉及条带")
            base = bytes((i * 37 + 11) & 0xFF for i in range(CHUNK * 4 + 999))
            pid = upload(cli, base, "recorder.bin")
            before_meta = cli.get("/api/files").json()["files"]
            before_row = next(f for f in before_meta if f["id"] == pid)
            check("新文件版本从 v1 起", before_row["version"] == 1)
            check("初始摘要已计算", len(before_row["sha256"]) == 64)
            old_sha = before_row["sha256"]
            blocks_before = all_block_bytes(vol())

            # 1) patch fully inside one stripe
            p1 = bytes(range(256)) * 4  # 1024 bytes
            r = patch(cli, pid, 100, p1)
            check("条带内补丁 200", r.status_code == 200, r.text)
            expected1 = patched(base, 100, p1)
            row1 = next(f for f in cli.get("/api/files").json()["files"]
                        if f["id"] == pid)
            check("条带内补丁内容正确", download(cli, pid) == expected1)
            check("文件编号保留", row1["id"] == pid)
            check("文件长度不变", row1["size"] == len(base))
            check("版本递增到 v2", row1["version"] == 2)
            check("摘要随覆盖同步更新",
                  row1["sha256"] == hashlib.sha256(expected1).hexdigest()
                  and row1["sha256"] != old_sha)

            # 2) patch ending exactly at EOF
            p2 = b"\x5a" * 500
            r = patch(cli, pid, len(base) - 500, p2)
            check("贴尾补丁 200", r.status_code == 200, r.text)
            expected2 = patched(expected1, len(base) - 500, p2)
            check("贴尾补丁内容正确", download(cli, pid) == expected2)

            # 3) cross-stripe patch spanning three stripes, misaligned
            off3 = CHUNK - 300
            p3 = os.urandom(CHUNK * 2 + 777)  # crosses two boundaries
            r = patch(cli, pid, off3, p3)
            check("跨条带补丁 200", r.status_code == 200, r.text)
            expected3 = patched(expected2, off3, p3)
            row3 = next(f for f in cli.get("/api/files").json()["files"]
                        if f["id"] == pid)
            check("跨条带补丁内容正确", download(cli, pid) == expected3)
            check("跨条带后版本 v4", row3["version"] == 4)
            check("跨条带后摘要正确",
                  row3["sha256"] == hashlib.sha256(expected3).hexdigest())
            check("条带数与长度保持",
                  row3["stripe_count"] == before_row["stripe_count"]
                  and row3["size"] == len(base))

            # 4) only touched stripes were rewritten
            blocks_after = all_block_bytes(vol())

            def stripes_hit(off: int, length: int) -> set[int]:
                return set(range(off // CHUNK, (off + length - 1) // CHUNK + 1))

            touched = (stripes_hit(100, len(p1))
                       | stripes_hit(len(base) - 500, len(p2))
                       | stripes_hit(off3, len(p3)))
            changed = [k for k in blocks_before
                       if blocks_after.get(k) != blocks_before[k]]
            changed_stripes = {
                int(Path(k[1]).stem.split("_")[1]) for k in changed
            }
            check("仅补丁涉及条带的块被重写",
                  changed_stripes == touched,
                  f"changed={sorted(changed_stripes)} touched={sorted(touched)}")
            untouched = [
                k for k in blocks_before
                if int(Path(k[1]).stem.split("_")[1]) not in touched
            ]
            check("未涉及条带三盘字节完全不变",
                  all(blocks_after[k] == blocks_before[k] for k in untouched))
            check("块总数不变（不新增条带）",
                  len(blocks_after) == len(blocks_before))

            # ---- patch validation & three-disk gating ------------------
            section("补丁校验：越界拒绝 / 空补丁 / 降级拒绝")
            r = patch(cli, pid, len(base) - 9, b"\x01" * 10)
            check("越界补丁 400 patch_out_of_bounds",
                  r.status_code == 400
                  and r.json()["error"] == "patch_out_of_bounds", r.text)
            r = patch(cli, pid, 0, b"")
            check("空补丁 400", r.status_code == 400, r.text)
            r = patch(cli, pid, -5, b"x")
            check("负偏移 422/400 被拒", r.status_code in (400, 422), r.text)
            check("被拒补丁不改内容", download(cli, pid) == expected3)
            check("被拒补丁不增版本",
                  next(f for f in cli.get("/api/files").json()["files"]
                       if f["id"] == pid)["version"] == 4)

            cli.post("/api/disks/0/offline").raise_for_status()
            r = patch(cli, pid, 10, b"\x02" * 8)
            check("盘离线时覆盖被拒(503)", r.status_code == 503, r.text)
            check("拒绝后文件仍是旧内容", download(cli, pid) == expected3)
            check("拒绝后无残留回滚目录",
                  not (DB_PATH.parent / "overwrite-rollback").exists()
                  or not any((DB_PATH.parent / "overwrite-rollback").iterdir()))
            cli.post("/api/disks/0/replace").raise_for_status()
            check("门控场景重建成功", wait_rebuild().get("state") == "done")

            r = patch(cli, "deadbeef" * 4, 0, b"x")
            check("未知文件补丁 404", r.status_code == 404, r.text)

            # ---- download / overwrite interleaving ----------------------
            section("覆盖与下载交错：每次下载都是完整旧版或完整新版")
            pre = download(cli, pid)
            inter_patch = os.urandom(CHUNK + 5000)
            inter_off = CHUNK - 100
            post = patched(pre, inter_off, inter_patch)
            results: list[bytes] = []
            errors: list[str] = []

            def downloader() -> None:
                try:
                    with httpx.Client(base_url=BASE, timeout=30) as c:
                        for _ in range(8):
                            results.append(download(c, pid))
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))

            threads = [threading.Thread(target=downloader) for _ in range(4)]
            for t in threads:
                t.start()
            time.sleep(0.05)
            r = patch(cli, pid, inter_off, inter_patch)
            check("交错覆盖成功", r.status_code == 200, r.text)
            for t in threads:
                t.join(timeout=30)
            check("下载线程无错误", errors == [], str(errors))
            check("所有下载结果非空", len(results) >= 16 and all(results))
            check("每次下载要么完整旧内容要么完整新内容",
                  all(c == pre or c == post for c in results),
                  f"mixed={sum(1 for c in results if c not in (pre, post))}")
            check("最终内容为新版", download(cli, pid) == post)

            # ---- write failure mid-overwrite: full rollback -------------
            section("写盘失败：未提交覆盖整体回滚")
            before_fail = download(cli, pid)
            volmod._OVERWRITE_IOFAIL_AFTER = 0
            fail_patch = b"\x7f" * (CHUNK * 2)
            r = patch(cli, pid, 10, fail_patch)
            volmod._OVERWRITE_IOFAIL_AFTER = -1
            check("写盘失败返回 500", r.status_code == 500, r.text)
            check("回滚后文件恢复 confirmed",
                  vol().db.get_file(pid)["status"] == "confirmed")
            check("回滚后内容与覆盖前完全一致",
                  download(cli, pid) == before_fail)
            job = vol().db.last_overwrite_job(pid)
            check("失败任务落库且含错误信息",
                  job["state"] == "failed" and bool(job["error"]))
            check("回滚目录已清理",
                  not (DB_PATH.parent / "overwrite-rollback").exists()
                  or not any((DB_PATH.parent / "overwrite-rollback").iterdir()))
            # second overwrite after a rolled-back one must work normally
            r = patch(cli, pid, 10, b"\x33" * 128)
            check("回滚后再次覆盖成功", r.status_code == 200, r.text)
            check("再次覆盖内容正确",
                  download(cli, pid) == patched(before_fail, 10, b"\x33" * 128))
            good_after_retry = download(cli, pid)

            # ---- service restart while overwrite is in flight -----------
            section("覆盖中服务重启：未提交覆盖不生效")
            crash_patch = b"\x9d" * (CHUNK * 2 + 333)
            crash_off = 5
            pre_restart = download(cli, pid)
            gate = threading.Event()
            volmod._OVERWRITE_PAUSE = gate

            def crashing_patch() -> None:
                # The BaseException escapes the normal handler; the thread
                # dies exactly like a killed process mid-overwrite.
                try:
                    vol().apply_patch(pid, crash_off, crash_patch)
                except BaseException:  # noqa: BLE001
                    pass

            th = threading.Thread(target=crashing_patch, daemon=True)
            th.start()
            # Wait until the first stripe has been physically rewritten.
            deadline = time.time() + 10
            while time.time() < deadline:
                jobs = vol().db.running_overwrite_jobs()
                if jobs and jobs[0]["stage"] == "prepared":
                    break
                time.sleep(0.01)
            check("覆盖确实进入条带重写阶段",
                  bool(vol().db.running_overwrite_jobs())
                  and vol().db.running_overwrite_jobs()[0]["stage"]
                  == "prepared")
            # Detach the module-level gate BEFORE the restart so the new
            # Volume instance does not park on the old thread's event.
            volmod._OVERWRITE_PAUSE = None
            # The worker is parked on the gate; restart underneath it.
            restart_service()
            with client() as cli2:
                meta_r = vol().db.get_file(pid)
                check("重启后文件恢复 confirmed", meta_r["status"] == "confirmed")
                check("重启后未提交覆盖不生效（内容为旧版）",
                      download(cli2, pid) == pre_restart)
                running = vol().db.running_overwrite_jobs()
                check("无悬挂覆盖任务", running == [])
                check("回滚目录已清空",
                      not (DB_PATH.parent / "overwrite-rollback").exists()
                      or not any(
                          (DB_PATH.parent / "overwrite-rollback").iterdir()))
                # a fresh overwrite after recovery works and bumps version
                r = patch(cli2, pid, 7, b"\x61" * 64)
                check("恢复后覆盖正常", r.status_code == 200, r.text)
                check("恢复后覆盖内容正确",
                      download(cli2, pid)
                      == patched(pre_restart, 7, b"\x61" * 64))
            volmod._OVERWRITE_PAUSE = None
            gate.set()  # release the parked old worker; it then exits
            new_version_content = download(client(), pid)

            # ---- degraded download of the new version -------------------
            with client() as cli2:
                section("覆盖成功后单盘离线仍可下载新版")
                for dead in range(3):
                    cli2.post(f"/api/disks/{dead}/offline").raise_for_status()
                    check(f"disk{dead} 离线仍可下载覆盖后新版",
                          download(cli2, pid) == new_version_content)
                    cli2.post(f"/api/disks/{dead}/replace").raise_for_status()
                    check(f"disk{dead} 重建成功",
                          wait_rebuild().get("state") == "done")
                    check(f"重建后新版内容不变",
                          download(cli2, pid) == new_version_content)

                # overwrite a second file and verify parity of rewritten stripes
                second = os.urandom(CHUNK * 2 + 10)
                sid = upload(cli2, second, "second.bin")
                r = patch(cli2, sid, CHUNK - 5, b"\xc3" * 40)
                check("第二文件覆盖成功", r.status_code == 200, r.text)
                parity_ok = True
                for s in vol().db.get_stripes(sid):
                    blocks = {}
                    for role in ("d0", "d1", "p"):
                        b = vol().db.get_block(sid, s["stripe_index"], role)
                        blocks[role] = storage.read_block(
                            vol().roots[b["disk_index"]], b["relpath"])
                    if any(coding.xor_bytes(
                            coding.xor_bytes(blocks["d0"], blocks["d1"]),
                            blocks["p"])):
                        parity_ok = False
                check("覆盖后条带仍满足 d0 XOR d1 XOR p = 0", parity_ok)
                check("第二文件覆盖内容正确",
                      download(cli2, sid) == patched(second, CHUNK - 5,
                                                     b"\xc3" * 40))
            _ = good_after_retry  # retained for readability of flow above

            # ---- interrupted upload: unconfirmed file invisible --------
            section("上传中断 / 崩溃恢复")
            interrupted = os.urandom(200_000)
            fake_id = "crashed00000000000000000000000000"
            vol().db.create_file({"id": fake_id, "name": "half.bin", "size": 0,
                                  "chunk_size": CHUNK, "stripe_count": 0})
            vol()._write_stripe(fake_id, 0, interrupted[:CHUNK])
            vol()._write_stripe(fake_id, 1, interrupted[CHUNK:2 * CHUNK])
            check("中断文件未确认、不在列表",
                  all(f["id"] != fake_id
                      for f in cli.get("/api/files").json()["files"]))
            check("中断文件残留块存在于盘上（待恢复清理）",
                  sum(1 for _ in _TMP.rglob(f"{fake_id}*")) > 0)

        # ---- service restart ------------------------------------------
        section("服务重启恢复")
        restart_service()
        with client() as cli:
            names = {f["name"] for f in cli.get("/api/files").json()["files"]}
            check("重启后未确认文件不可见", "half.bin" not in names)
            check("重启后已确认文件仍可见",
                  {"alpha.bin", "beta.bin", "empty.bin"}.issubset(names))
            check("重启清掉了未确认文件的残留块",
                  sum(1 for _ in _TMP.rglob(f"{fake_id}*")) == 0)
            check("已确认文件重启后仍可读",
                  download(cli, id_a) == payload_a
                  and download(cli, id_b) == payload_b)
            check("重启后无悬挂重建任务", vol().status()["rebuild"] is None)

        # ---- rebuild interrupted by restart, then resumed --------------
        section("重建中断后自动续建")
        volmod._REBUILD_STRIPE_DELAY = 0.03  # make the rebuild interruptible
        vol().take_disk_offline(0)
        vol().replace_disk(0)
        time.sleep(0.1)  # a few stripes rebuilt, then "process" restarts
        restart_service()
        with client() as cli:
            st = vol().status()
            check("重启后重建任务自动续跑",
                  st["rebuild"] is not None
                  and st["rebuild"]["state"] == "running")
            job = wait_rebuild()
            check("续建最终成功", job.get("state") == "done", str(job))
            check("续建后磁盘 online", vol().db.disk_state(0) == "online")
            check("续建后全部文件可读",
                  download(cli, id_a) == payload_a
                  and download(cli, id_b) == payload_b)
            check("续建后无缺失条带", vol()._stripes_missing_on_disk(0) == [])
        volmod._REBUILD_STRIPE_DELAY = 0

        # ---- upload refused when a disk is offline --------------------
        section("降级状态下拒绝产生残缺条带")
        vol().take_disk_offline(2)
        with client() as cli:
            r = cli.post("/api/files",
                         files={"file": ("x.bin", io.BytesIO(b"hi"), "text/plain")})
            check("盘离线时上传被拒(503)", r.status_code == 503, r.text)
            check("失败上传不留可见文件",
                  all(f["name"] != "x.bin"
                      for f in cli.get("/api/files").json()["files"]))
            vol().replace_disk(2)
            check("恢复重建成功", wait_rebuild().get("state") == "done")
            fid = upload(cli, b"final-check", "final.bin")
            check("恢复冗余后上传正常", download(cli, fid) == b"final-check")

        # ---- physical disk loss: directory vanishes --------------------
        section("物理盘目录消失（重启后识别为下线）")
        payload_c = os.urandom(120_000)
        id_c = upload(client(), payload_c, "gamma.bin")
        vol().shutdown()
        lost = DATA_DIR / "disk1"
        shutil.move(str(lost), str(_TMP / "disk1-ripped-out"))
        webapp.volume = Volume(DATA_DIR, DB_PATH, chunk_size=CHUNK)
        webapp.volume.start()
        with client() as cli:
            st = cli.get("/api/status").json()
            check("缺失目录的盘被标记 offline",
                  next(d for d in st["disks"] if d["disk_index"] == 1)["state"]
                  == "offline")
            check("目录缺失时文件仍可下载（其余两块盘重建）",
                  download(cli, id_c) == payload_c)
            # operator replaces the disk; service provisions a fresh dir
            r = cli.post("/api/disks/1/replace")
            check("换盘接口自动重建空目录并启动重建",
                  r.status_code == 200, r.text)
            check("物理换盘后重建成功",
                  wait_rebuild().get("state") == "done")
            check("物理换盘后文件可读", download(cli, id_c) == payload_c)

        # ---- interleaved upload + rebuild interrupted by restart --------
        section("重建中上传，随后重启：新条带不遗漏")
        volmod._REBUILD_STRIPE_DELAY = 0.02
        vol().take_disk_offline(0)
        vol().replace_disk(0)
        time.sleep(0.05)
        with client() as cli:
            mid_payload = os.urandom(140_000)
            mid_id = upload(cli, mid_payload, "mid-rebuild.bin")
            check("重建中途上传成功且可读",
                  download(cli, mid_id) == mid_payload)
        restart_service()
        with client() as cli:
            check("重启后续建任务在跑",
                  vol().status()["rebuild"] is not None
                  and vol().status()["rebuild"]["state"] == "running")
            check("续建成功", wait_rebuild().get("state") == "done")
            check("中途上传的文件续建后可读",
                  download(cli, mid_id) == mid_payload)
            check("中途文件在新盘上无缺失条带",
                  vol()._stripes_missing_on_disk(0) == [])
            # another failure must still be survivable for the new file
            vol().take_disk_offline(2)
            check("再坏一盘，中途文件仍可下载",
                  download(cli, mid_id) == mid_payload)
            vol().replace_disk(2)
            check("最终重建成功", wait_rebuild().get("state") == "done")
        volmod._REBUILD_STRIPE_DELAY = 0
    finally:
        vol().shutdown()
        server.should_exit = True
        thread.join(timeout=10)
        shutil.rmtree(_TMP, ignore_errors=True)

    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
