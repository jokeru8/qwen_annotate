"""Bounded local HTTP interface for visual review of one workspace."""

from __future__ import annotations

import os
import stat
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .review import (
    InvalidManualAnnotation,
    ManualDecision,
    ManualDecisionConflict,
    ReviewServices,
    _assert_current_source,
    _candidate_annotation,
    _validate_dataset,
    _validate_record_run_context,
    _workspace_root,
    apply_manual_decision,
)
from .workspace import EpisodeRecord, WorkspaceStore


ReviewStatus = Literal[
    "pending", "coarse_done", "refine_done", "accepted", "needs_review", "failed"
]


class ReviewRuntime:
    """Read-only view adapter pinned to one validated workspace and dataset."""

    def __init__(self, work_dir: Path, *, services: ReviewServices | None = None) -> None:
        self.root = _workspace_root(work_dir)
        self.services = services or ReviewServices()
        self.store = WorkspaceStore(self.root, clock=self.services.clock)
        self.manifest, self.config = self.store.load_manifest_with_provenance()
        self.dataset = self.services.inspect_dataset(self.config)
        _validate_dataset(self.manifest, self.dataset)

    def records(self) -> list[EpisodeRecord]:
        return [
            self.store.load_episode(index) for index in range(self.manifest.total_episodes)
        ]

    def record(self, index: int) -> EpisodeRecord:
        if type(index) is not int or not 0 <= index < self.manifest.total_episodes:
            raise KeyError(index)
        record = self.store.load_episode(index)
        _validate_record_run_context(record, self.manifest)
        _assert_current_source(self.manifest, self.dataset, record)
        return record

    def session_payload(self) -> dict[str, object]:
        counts = Counter(record.status for record in self.records())
        return {
            "mode": self.manifest.mode,
            "fps": self.manifest.fps,
            "camera_keys": list(self.manifest.camera_keys),
            "primary_camera": self.config.primary_camera,
            "high_level_instruction": self.manifest.high_level_instruction,
            "subtasks": [item.model_dump(mode="json") for item in self.manifest.subtasks],
            "total_episodes": self.manifest.total_episodes,
            "min_segment_frames": self.manifest.min_segment_frames,
            "status_counts": dict(counts),
        }

    def episode_summary(self, record: EpisodeRecord) -> dict[str, object]:
        return {
            "episode_index": record.episode_index,
            "status": record.status,
            "decision_source": record.decision_source,
            "updated_at": record.updated_at.isoformat(),
            "failure_category": record.failure_category,
            "review_reasons": list(record.review_reasons),
        }

    def episode_payload(self, index: int) -> dict[str, object]:
        record = self.record(index)
        episode = self.dataset.episodes[index]
        return {
            **self.episode_summary(record),
            "episode_length": episode.length,
            "task": episode.task,
            "source_fingerprint": record.source_fingerprint,
            "run_fingerprint": record.run_fingerprint,
            "mode": self.manifest.mode,
            "candidate_annotation": _candidate_annotation(record),
            "validation_issues": [
                item.model_dump(mode="json") for item in record.validation_issues
            ],
            "coarse_attempts": [item.model_dump(mode="json") for item in record.coarse_attempts],
            "refine_attempts": [item.model_dump(mode="json") for item in record.refine_attempts],
            "videos": {
                camera: {
                    "url": f"/api/episodes/{index}/videos/{quote(camera, safe='')}",
                    "from_timestamp": ref.from_timestamp,
                    "to_timestamp": ref.to_timestamp,
                }
                for camera, ref in episode.videos.items()
            },
        }

    def open_video(self, index: int, camera: str) -> tuple[int, int]:
        record = self.record(index)
        episode = self.dataset.episodes[index]
        if camera not in self.manifest.camera_keys or camera not in episode.videos:
            raise KeyError(camera)
        path = episode.videos[camera].path
        if path.is_symlink():
            raise ValueError("video is not a regular file")
        resolved = path.resolve()
        try:
            resolved.relative_to(self.manifest.dataset_root.resolve())
        except ValueError:
            raise ValueError("video is outside the dataset") from None
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise ValueError("video is not a regular file")
        try:
            _assert_current_source(self.manifest, self.dataset, record)
            path_info = os.stat(resolved, follow_symlinks=False)
            identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(getattr(info, field) != getattr(path_info, field) for field in identity):
                raise ValueError("video changed while opening")
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor, info.st_size


def _parse_range(value: str, size: int) -> tuple[int, int]:
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("invalid byte range")
    raw = value.removeprefix("bytes=")
    if raw.count("-") != 1:
        raise ValueError("invalid byte range")
    left, right = raw.split("-", 1)
    if left:
        if not left.isascii() or not left.isdecimal():
            raise ValueError("invalid byte range")
        start = int(left)
        if start >= size:
            raise ValueError("unsatisfiable byte range")
        if right:
            if not right.isascii() or not right.isdecimal():
                raise ValueError("invalid byte range")
            end = min(int(right), size - 1)
            if end < start:
                raise ValueError("invalid byte range")
        else:
            end = size - 1
        return start, end
    if not right.isascii() or not right.isdecimal() or int(right) <= 0:
        raise ValueError("invalid byte range")
    length = min(int(right), size)
    return size - length, size - 1


def _fd_chunks(descriptor: int, start: int, length: int) -> Iterator[bytes]:
    try:
        os.lseek(descriptor, start, os.SEEK_SET)
        remaining = length
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise OSError("video changed while streaming")
            remaining -= len(chunk)
            yield chunk
    finally:
        os.close(descriptor)


def create_review_app(
    work_dir: Path, *, services: ReviewServices | None = None
) -> FastAPI:
    runtime = ReviewRuntime(work_dir, services=services)
    app = FastAPI(title="Robo-annotate Studio", docs_url=None, redoc_url=None)
    app.state.review_runtime = runtime
    web_root = Path(__file__).parent / "review_web"
    csp = "default-src 'self'; img-src 'self'; media-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"

    @app.middleware("http")
    async def bound_decision_body(request: Request, call_next):
        if request.method == "POST" and request.url.path.endswith("/decision"):
            raw_length = request.headers.get("content-length")
            if raw_length is None or not raw_length.isascii() or not raw_length.isdecimal():
                return JSONResponse(status_code=411, content={"detail": "content length required"})
            if int(raw_length) > 128 * 1024:
                return JSONResponse(status_code=413, content={"detail": "decision body too large"})
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def index_page():
        return FileResponse(
            web_root / "index.html",
            media_type="text/html",
            headers={"Content-Security-Policy": csp, "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str):
        media_types = {"app.js": "text/javascript", "style.css": "text/css"}
        if name not in media_types:
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(
            web_root / name,
            media_type=media_types[name],
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/session")
    def session() -> dict[str, object]:
        return runtime.session_payload()

    @app.get("/api/episodes")
    def episodes(status: ReviewStatus | None = Query(default=None)) -> list[dict[str, object]]:
        records = runtime.records()
        if status is not None:
            records = [record for record in records if record.status == status]
        return [runtime.episode_summary(record) for record in records]

    @app.get("/api/episodes/{index}")
    def episode(index: int) -> dict[str, object]:
        try:
            return runtime.episode_payload(index)
        except KeyError:
            raise HTTPException(status_code=404, detail="episode not found") from None
        except ValueError:
            raise HTTPException(status_code=409, detail="episode source or state changed") from None

    @app.get("/api/episodes/{index}/videos/{camera:path}")
    def video(index: int, camera: str, range_header: str | None = Header(None, alias="Range")):
        try:
            descriptor, size = runtime.open_video(index, camera)
        except KeyError:
            raise HTTPException(status_code=404, detail="video not found") from None
        except ValueError:
            raise HTTPException(status_code=409, detail="video source changed") from None
        headers = {"Accept-Ranges": "bytes"}
        if range_header is None:
            headers["Content-Length"] = str(size)
            return StreamingResponse(
                _fd_chunks(descriptor, 0, size), media_type="video/mp4", headers=headers,
            )
        try:
            start, end = _parse_range(range_header, size)
        except ValueError:
            os.close(descriptor)
            raise HTTPException(
                status_code=416,
                detail="range not satisfiable",
                headers={"Content-Range": f"bytes */{size}"},
            ) from None
        length = end - start + 1
        headers.update({
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(length),
        })
        return StreamingResponse(
            _fd_chunks(descriptor, start, length),
            status_code=206,
            media_type="video/mp4",
            headers=headers,
        )

    @app.post("/api/episodes/{index}/decision")
    def save_decision(index: int, decision: ManualDecision) -> dict[str, object]:
        if index != decision.episode_index:
            raise HTTPException(
                status_code=409,
                detail={"code": "episode_mismatch", "message": "episode identity mismatch"},
            )
        try:
            apply_manual_decision(runtime.root, decision, services=runtime.services)
        except InvalidManualAnnotation as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": exc.code,
                    "message": "annotation violates workspace constraints",
                    "issues": exc.issues,
                },
            ) from None
        except ManualDecisionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": exc.code, "message": str(exc)},
            ) from None
        except ValueError:
            raise HTTPException(
                status_code=409,
                detail={"code": "workspace_conflict", "message": "workspace state changed"},
            ) from None
        return runtime.episode_payload(index)

    return app
