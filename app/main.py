"""FastAPI application: chunked, resumable, hash-verified uploads + downloads."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from . import config, db, storage

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Guards against concurrent merges of the same session.
_merge_locks: dict[str, threading.Lock] = {}
_merge_locks_guard = threading.Lock()


def _merge_lock(session_id: str) -> threading.Lock:
    with _merge_locks_guard:
        return _merge_locks.setdefault(session_id, threading.Lock())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-initialise storage on every boot; existing state is preserved,
    # so interrupted sessions survive restarts.
    config.ensure_dirs()
    db.init_db()
    storage.reconcile()
    yield


app = FastAPI(title="Chunked Upload Service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- schemas


class CreateSessionRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    total_size: int = Field(gt=0, description="Total file size in bytes")
    chunk_size: int = Field(gt=0, le=config.MAX_CHUNK_SIZE)
    file_sha256: str | None = Field(
        default=None, description="Optional SHA-256 of the whole file, checked at merge"
    )

    @field_validator("file_sha256")
    @classmethod
    def _check_digest(cls, value: str | None) -> str | None:
        if value is not None:
            if not SHA256_RE.match(value):
                raise ValueError("file_sha256 must be 64 hex characters")
            return value.lower()
        return None


# ---------------------------------------------------------------- helpers


def _get_session_or_404(session_id: str):
    with db.connect() as conn:
        session = db.get_session(conn, session_id)
        if session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
        return session


_SINGLE_RANGE_RE = re.compile(r"bytes=\s*(\d*)\s*-\s*(\d*)", re.IGNORECASE)


def _parse_single_range(header: str, size: int) -> tuple[int, int]:
    """Parse one `bytes` range into an inclusive (start, end) pair.

    Multi-range sets, malformed syntax and unsatisfiable ranges all raise
    ValueError; the endpoint maps that to 416.
    """
    value = header.strip()
    if "," in value:
        raise ValueError("multiple ranges are not supported")
    match = _SINGLE_RANGE_RE.fullmatch(value)
    if match is None:
        raise ValueError("unsupported Range syntax")
    first, last = match.groups()
    if not first and not last:
        raise ValueError("empty range specifier")
    if first:
        start = int(first)
        end = int(last) if last else size - 1
        if start > end:
            raise ValueError("range start is after range end")
        if start >= size:
            raise ValueError("range start is beyond the end of the file")
        return start, min(end, size - 1)
    suffix = int(last)
    if suffix == 0:
        raise ValueError("zero-length suffix range")
    return max(size - suffix, 0), size - 1


def _etag_list_matches(header: str, etag: str) -> bool:
    """If-None-Match uses weak comparison; `*` matches any representation."""
    for candidate in header.split(","):
        candidate = candidate.strip()
        if candidate.startswith("W/"):
            candidate = candidate[2:].strip()
        if candidate == "*" or candidate == etag:
            return True
    return False


def _etag_strong_match(header: str, etag: str) -> bool:
    """If-Range requires a strong comparison; weak tags and dates never match."""
    candidate = header.strip()
    return not candidate.startswith("W/") and candidate == etag


# ---------------------------------------------------------------- endpoints


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/uploads", status_code=status.HTTP_201_CREATED)
def create_session(body: CreateSessionRequest) -> dict:
    """Create an upload session. Chunk count is derived from size/chunk_size."""
    total_chunks = math.ceil(body.total_size / body.chunk_size)
    session_id = uuid.uuid4().hex
    now = db.utcnow()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, filename, total_size, chunk_size, "
            "total_chunks, file_sha256, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'uploading', ?, ?)",
            (
                session_id,
                body.filename,
                body.total_size,
                body.chunk_size,
                total_chunks,
                body.file_sha256,
                now,
                now,
            ),
        )
        session = db.get_session(conn, session_id)
        return db.session_payload(conn, session)


@app.get("/uploads")
def list_sessions() -> dict:
    """List all sessions (most recent first) with progress counters."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT s.*, "
            "       (SELECT COUNT(*) FROM chunks c WHERE c.session_id = s.id) "
            "       AS received "
            "FROM sessions s ORDER BY s.created_at DESC LIMIT 200"
        ).fetchall()
    return {
        "sessions": [
            {
                "session_id": row["id"],
                "filename": row["filename"],
                "status": row["status"],
                "total_chunks": row["total_chunks"],
                "received_count": row["received"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    }


@app.get("/uploads/{session_id}")
def get_session(session_id: str) -> dict:
    """Session status, including the list of missing chunk indexes."""
    with db.connect() as conn:
        session = db.get_session(conn, session_id)
        if session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
        return db.session_payload(conn, session)


@app.put("/uploads/{session_id}/chunks/{index}")
async def upload_chunk(
    session_id: str,
    index: int,
    request: Request,
    x_chunk_sha256: str | None = Header(default=None),
    sha256: str | None = Query(default=None),
) -> dict:
    """Receive one chunk (raw body). Out-of-order and repeat uploads are fine.

    The chunk's SHA-256 must be supplied via the X-Chunk-SHA256 header or the
    ?sha256= query parameter; mismatched or wrongly sized payloads are rejected
    and nothing is persisted, so clients can simply retry.
    """
    digest = (x_chunk_sha256 or sha256 or "").lower()
    if not SHA256_RE.match(digest):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "provide the chunk SHA-256 via X-Chunk-SHA256 header or ?sha256=",
        )

    with db.connect() as conn:
        session = db.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    if session["status"] != "uploading":
        raise HTTPException(status.HTTP_409_CONFLICT, "session already completed")
    if not 0 <= index < session["total_chunks"]:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"chunk index must be in [0, {session['total_chunks'] - 1}]",
        )

    expected_size = storage.expected_chunk_size(session, index)
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) != expected_size:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"chunk {index} must be exactly {expected_size} bytes",
        )

    # Stream to a temp file while hashing; publish atomically on success.
    cdir = storage.chunks_dir(session_id)
    cdir.mkdir(parents=True, exist_ok=True)
    tmp_path = cdir / f"{index:08d}.part.tmp-{uuid.uuid4().hex}"
    hasher = hashlib.sha256()
    size = 0
    try:
        with open(tmp_path, "wb") as out:
            async for piece in request.stream():
                hasher.update(piece)
                out.write(piece)
                size += len(piece)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    if size != expected_size:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"chunk {index} must be exactly {expected_size} bytes, got {size}",
        )
    if hasher.hexdigest() != digest:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "chunk sha256 mismatch")

    os.replace(tmp_path, storage.chunk_path(session_id, index))

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO chunks (session_id, chunk_index, size, sha256, received_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (session_id, chunk_index) DO UPDATE SET "
            "size = excluded.size, sha256 = excluded.sha256, "
            "received_at = excluded.received_at",
            (session_id, index, size, digest, db.utcnow()),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (db.utcnow(), session_id),
        )
        session = db.get_session(conn, session_id)
        return db.session_payload(conn, session)


@app.post("/uploads/{session_id}/complete")
def complete_session(session_id: str) -> dict:
    """Merge all chunks into the final file. Idempotent.

    Fails with 409 (and the missing list) until every chunk is present; each
    chunk's hash is re-verified during the merge, and the target file is
    published atomically via rename.
    """
    lock = _merge_lock(session_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "a merge is already in progress for this session"
        )
    try:
        try:
            return storage.merge_session(session_id)
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
        except storage.MergeError as exc:
            code = (
                status.HTTP_409_CONFLICT
                if exc.missing
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(
                code,
                detail={"message": str(exc), "missing_chunks": exc.missing},
            )
    finally:
        lock.release()


@app.api_route("/uploads/{session_id}/content", methods=["GET", "HEAD"])
def download_content(
    session_id: str,
    request: Request,
    range_header: str | None = Header(default=None, alias="range"),
    if_none_match: str | None = Header(default=None),
    if_range: str | None = Header(default=None),
):
    """Download the merged file, in full or as a single byte range.

    Always emits Accept-Ranges and a strong ETag built from the file's SHA-256.
    If-None-Match short-circuits to 304; If-Range gates whether a Range is
    honoured (mismatch → full 200). Multi-range sets, bad syntax and
    unsatisfiable ranges get 416 with `Content-Range: bytes */<size>`. Payload
    bytes are streamed from disk in bounded reads, never buffered wholesale.
    """
    with db.connect() as conn:
        session = db.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    if session["status"] != "completed":
        raise HTTPException(status.HTTP_409_CONFLICT, "session is not completed yet")

    path = Path(session["final_path"])
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "merged file is missing")

    size = path.stat().st_size
    etag = f'"{session["file_sha256"]}"'
    headers = {
        "Accept-Ranges": "bytes",
        "ETag": etag,
        "Content-Disposition": (
            f'attachment; filename="{storage.sanitize_filename(session["filename"])}"'
        ),
    }

    # Preconditions are evaluated before Range (RFC 9110 §13.2.2).
    if if_none_match is not None and _etag_list_matches(if_none_match, etag):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)

    start, length, code = 0, size, status.HTTP_200_OK
    if range_header is not None and (
        if_range is None or _etag_strong_match(if_range, etag)
    ):
        try:
            start, end = _parse_single_range(range_header, size)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                str(exc),
                headers={
                    "Accept-Ranges": "bytes",
                    "ETag": etag,
                    "Content-Range": f"bytes */{size}",
                },
            )
        length = end - start + 1
        code = status.HTTP_206_PARTIAL_CONTENT
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"

    headers["Content-Length"] = str(length)
    if request.method == "HEAD":  # same metadata as GET, but no payload body
        return Response(status_code=code, headers=headers)
    return StreamingResponse(
        storage.iter_file(path, start, length),
        status_code=code,
        headers=headers,
        media_type="application/octet-stream",
    )


@app.delete("/uploads/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def abort_session(session_id: str) -> None:
    """Abort a session: delete its chunks, metadata and (if any) merged file."""
    with db.connect() as conn:
        session = db.get_session(conn, session_id)
        if session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    shutil.rmtree(storage.session_dir(session_id), ignore_errors=True)
    if session["final_path"]:
        try:
            os.unlink(session["final_path"])
        except OSError:
            pass
