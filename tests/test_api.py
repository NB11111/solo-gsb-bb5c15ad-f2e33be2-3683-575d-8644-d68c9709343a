"""End-to-end tests for the chunked upload service.

Run with:  DATA_DIR=$(mktemp -d) pytest tests/
"""

from __future__ import annotations

import hashlib
import os
import tempfile

# Point storage at a throwaway directory BEFORE importing the app.
_TMP = tempfile.mkdtemp(prefix="chunked-upload-test-")
os.environ["DATA_DIR"] = _TMP

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_file(chunk_size: int, total_chunks: int, last_size: int) -> tuple[bytes, list[bytes]]:
    chunks = []
    for i in range(total_chunks):
        size = last_size if i == total_chunks - 1 else chunk_size
        chunks.append(bytes(f"chunk-{i:05d}-".encode()) + os.urandom(size - 12))
    return b"".join(chunks), chunks


def _create(client: TestClient, data: bytes, chunk_size: int, **extra) -> dict:
    resp = client.post(
        "/uploads",
        json={
            "filename": "sample.bin",
            "total_size": len(data),
            "chunk_size": chunk_size,
            **extra,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _put(client: TestClient, sid: str, index: int, payload: bytes, digest: str | None = None):
    return client.put(
        f"/uploads/{sid}/chunks/{index}",
        content=payload,
        headers={"X-Chunk-SHA256": digest or _sha256(payload)},
    )


def test_full_flow_out_of_order_with_retry():
    with TestClient(app) as client:
        chunk_size = 1024
        data, chunks = _make_file(chunk_size, total_chunks=5, last_size=500)
        session = _create(client, data, chunk_size, file_sha256=_sha256(data))
        sid = session["session_id"]
        assert session["total_chunks"] == 5
        assert session["missing_chunks"] == [0, 1, 2, 3, 4]

        # Corrupted payload must be rejected and not persisted.
        bad = _put(client, sid, 0, chunks[0], digest="0" * 64)
        assert bad.status_code == 400
        assert client.get(f"/uploads/{sid}").json()["missing_chunks"] == [0, 1, 2, 3, 4]

        # Out-of-order upload: 3, 1, 4 ...
        for i in (3, 1, 4):
            resp = _put(client, sid, i, chunks[i])
            assert resp.status_code == 200, resp.text
        state = client.get(f"/uploads/{sid}").json()
        assert state["missing_chunks"] == [0, 2]
        assert state["received_count"] == 3

        # Retry of an already-received chunk is idempotent.
        assert _put(client, sid, 3, chunks[3]).status_code == 200
        assert client.get(f"/uploads/{sid}").json()["received_count"] == 3

        # Cannot merge while chunks are missing.
        early = client.post(f"/uploads/{sid}/complete")
        assert early.status_code == 409
        assert early.json()["detail"]["missing_chunks"] == [0, 2]

        # Finish the rest and merge.
        assert _put(client, sid, 0, chunks[0]).status_code == 200
        assert _put(client, sid, 2, chunks[2]).status_code == 200
        done = client.post(f"/uploads/{sid}/complete")
        assert done.status_code == 200, done.text
        body = done.json()
        assert body["status"] == "completed"
        assert body["file_sha256"] == _sha256(data)

        final_path = body["final_path"]
        assert os.path.isfile(final_path)
        with open(final_path, "rb") as fh:
            assert fh.read() == data

        # Complete is idempotent; further chunk uploads are refused.
        assert client.post(f"/uploads/{sid}/complete").status_code == 200
        assert _put(client, sid, 0, chunks[0]).status_code == 409


def test_restart_resumes_unfinished_session():
    chunk_size = 512
    data, chunks = _make_file(chunk_size, total_chunks=3, last_size=512)

    with TestClient(app) as client:
        sid = _create(client, data, chunk_size)["session_id"]
        assert _put(client, sid, 0, chunks[0]).status_code == 200
        assert _put(client, sid, 2, chunks[2]).status_code == 200

    # "Restart": a brand-new app instance over the same DATA_DIR.
    with TestClient(app) as client:
        state = client.get(f"/uploads/{sid}").json()
        assert state["status"] == "uploading"
        assert state["missing_chunks"] == [1]

        assert _put(client, sid, 1, chunks[1]).status_code == 200
        done = client.post(f"/uploads/{sid}/complete")
        assert done.status_code == 200, done.text
        with open(done.json()["final_path"], "rb") as fh:
            assert fh.read() == data


def test_validation_errors():
    with TestClient(app) as client:
        data, chunks = _make_file(256, total_chunks=2, last_size=100)
        sid = _create(client, data, 256)["session_id"]

        # Index out of range.
        assert _put(client, sid, 2, chunks[0]).status_code == 400
        # Missing digest.
        assert client.put(f"/uploads/{sid}/chunks/0", content=chunks[0]).status_code == 400
        # Wrong size.
        assert _put(client, sid, 0, chunks[0][:-1]).status_code == 400
        # Unknown session.
        assert client.get("/uploads/nope").status_code == 404
        # Bad overall digest must fail the merge with 422.
        wrong = _create(client, data, 256, file_sha256="f" * 64)
        wsid = wrong["session_id"]
        assert _put(client, wsid, 0, chunks[0]).status_code == 200
        assert _put(client, wsid, 1, chunks[1]).status_code == 200
        assert client.post(f"/uploads/{wsid}/complete").status_code == 422


def test_abort_session():
    with TestClient(app) as client:
        data, chunks = _make_file(256, total_chunks=2, last_size=256)
        sid = _create(client, data, 256)["session_id"]
        assert _put(client, sid, 0, chunks[0]).status_code == 200
        assert client.delete(f"/uploads/{sid}").status_code == 204
        assert client.get(f"/uploads/{sid}").status_code == 404
