"""Offline review artifacts and validated human acceptance decisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import AnnotationConfig
from .constraints import validate_annotation
from .lerobot import DatasetIndex, inspect_dataset
from .models import FinalAnnotation
from .video import FrameSample, extract_frames, uniform_indices, window_indices
from .workspace import EpisodeRecord, RunManifest, WorkspaceStore, compute_source_fingerprint


_MAX_DECISION_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_CANDIDATES = 64
_MAX_IMAGES_PER_EPISODE = 512
_OWNER = ".qwen-annotate-review-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HumanDecision(BaseModel):
    """Portable, strict human decision file written by the offline UI."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    episode_index: int = Field(ge=0)
    source_fingerprint: str
    start_subtask_index: int = Field(ge=0)
    boundaries: list[int]

    @field_validator("source_fingerprint")
    @classmethod
    def exact_fingerprint(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("source_fingerprint must be lowercase SHA-256")
        return value

    @field_validator("boundaries")
    @classmethod
    def strict_boundaries(cls, value: list[int]) -> list[int]:
        if any(type(item) is not int for item in value):
            raise ValueError("boundaries must contain strict integers")
        return value


@dataclass(frozen=True, slots=True)
class ReviewServices:
    inspect_dataset: Callable[[AnnotationConfig], DatasetIndex] = inspect_dataset
    sampler: Callable[[Path, str, list[int], float], list[FrameSample]] = extract_frames
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


def load_human_decision(path: Path) -> HumanDecision:
    """Read one bounded, regular, no-follow decision JSON file strictly."""
    payload = _read_regular(path, _MAX_DECISION_BYTES)
    return HumanDecision.model_validate(_strict_json(payload))


def render_review_site(work_dir: Path, *, services: ReviewServices | None = None) -> Path:
    """Render a self-contained static review bundle without modifying source data."""
    service = services or ReviewServices()
    root = _workspace_root(work_dir)
    store = WorkspaceStore(root)
    manifest = _load_manifest(root)
    config = _effective_config(manifest, root)
    dataset = service.inspect_dataset(config)
    _validate_dataset(manifest, dataset)

    records = [store.load_episode(index) for index in range(manifest.total_episodes)]
    review_records = sorted((record for record in records if record.status == "needs_review"),
                            key=lambda item: item.episode_index)
    for record in review_records:
        _assert_current_source(manifest, dataset, record)

    previews = root / "previews"
    destination = previews / "needs_review"
    _assert_directory(previews, "workspace previews")
    _assert_replaceable_destination(destination)
    staging = previews / f".needs_review.staging-{secrets.token_hex(12)}"
    staging.mkdir(mode=0o700)
    try:
        (staging / _OWNER).write_text("owned\n", encoding="ascii")
        shutil.copyfile(Path(__file__).parent / "static" / "review.js", staging / "review.js")
        episodes = []
        for record in review_records:
            episode = dataset.episodes[record.episode_index]
            episodes.append(_render_episode(staging, config, manifest, episode, record, service))
            _assert_current_source(manifest, dataset, record)
        environment = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            autoescape=select_autoescape(("html", "xml"), default=True),
            undefined=StrictUndefined,
        )
        html = environment.get_template("review.html.j2").render(
            episodes=episodes, high_level_instruction=manifest.high_level_instruction,
            subtasks=[item.model_dump() for item in manifest.subtasks],
        )
        _write_file(staging / "index.html", html.encode("utf-8"))
        _publish_directory(staging, destination)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return destination / "index.html"


def apply_human_decision(
    work_dir: Path,
    episode_index: int,
    annotation: FinalAnnotation | HumanDecision,
    *,
    source_fingerprint: str | None = None,
    services: ReviewServices | None = None,
) -> EpisodeRecord:
    """Validate and atomically transition one needs-review record to accepted."""
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer")
    service = services or ReviewServices()
    root = _workspace_root(work_dir)
    manifest = _load_manifest(root)
    if episode_index >= manifest.total_episodes:
        raise ValueError("episode identity is outside the workspace manifest")
    store = WorkspaceStore(root, clock=service.clock)
    record = store.load_episode(episode_index)
    if record.status != "needs_review":
        raise ValueError("human decisions require a needs_review record")

    if isinstance(annotation, HumanDecision):
        if annotation.episode_index != episode_index:
            raise ValueError("decision episode identity does not match requested episode")
        supplied_fingerprint = annotation.source_fingerprint
        final = FinalAnnotation(
            start_subtask_index=annotation.start_subtask_index,
            boundaries=list(annotation.boundaries),
        )
        if source_fingerprint is not None and source_fingerprint != supplied_fingerprint:
            raise ValueError("conflicting source fingerprints were supplied")
    elif isinstance(annotation, FinalAnnotation):
        supplied_fingerprint = source_fingerprint
        final = FinalAnnotation.model_validate(annotation.model_dump())
    else:
        raise TypeError("annotation must be FinalAnnotation or HumanDecision")
    if supplied_fingerprint is None:
        raise ValueError("source_fingerprint is required")
    if supplied_fingerprint != record.source_fingerprint:
        raise ValueError("decision source fingerprint does not match episode record")

    config = _effective_config(manifest, root)
    dataset = service.inspect_dataset(config)
    _validate_dataset(manifest, dataset)
    episode = dataset.episodes[episode_index]
    current_fingerprint = compute_source_fingerprint(manifest.dataset_root, episode)
    if current_fingerprint != record.source_fingerprint:
        raise ValueError("current source fingerprint does not match episode record")
    issues = validate_annotation(
        final, manifest.mode, len(manifest.subtasks), episode.length, manifest.min_segment_frames,
    )
    if issues:
        raise ValueError(", ".join(issue.code for issue in issues))

    # Recompute after validation, immediately before constructing/saving the transition.
    current_fingerprint = compute_source_fingerprint(manifest.dataset_root, episode)
    if current_fingerprint != supplied_fingerprint:
        raise ValueError("source fingerprint changed before human decision save")
    updated_at = _strict_time(service.clock(), record.updated_at)
    details = dict(record.sampling_details)
    audit = {
        "prior_reasons": list(record.review_reasons),
        "prior_validation_issues": [item.model_dump(mode="json") for item in record.validation_issues],
        "prior_candidate": _candidate_annotation(record),
        "accepted_annotation": final.model_dump(mode="json"),
        "supplied_source_fingerprint": supplied_fingerprint,
        "current_source_fingerprint": current_fingerprint,
        "timestamp": updated_at.isoformat(),
    }
    previous_audits = details.get("human_decisions", [])
    if not isinstance(previous_audits, list):
        raise ValueError("existing human decision audit is invalid")
    details["human_decisions"] = [*previous_audits, audit]
    _append_human_outbox(details, record, updated_at)
    accepted = record.model_copy(update={
        "status": "accepted",
        "final_annotation": final,
        "validation_issues": [],
        "review_reasons": [],
        "failure_category": None,
        "decision_source": "human",
        "sampling_details": details,
        "updated_at": updated_at,
    })
    accepted = EpisodeRecord.model_validate(accepted.model_dump())
    store.save_episode_transactional(accepted)
    return accepted


def _render_episode(staging, config, manifest, episode, record, service):
    episode_name = f"episode_{episode.episode_index:06d}"
    assets = staging / episode_name
    assets.mkdir()
    candidates_all = _candidate_frames(record, episode.length)
    candidate_queue = candidates_all[:_MAX_CANDIDATES]
    omitted = candidates_all[_MAX_CANDIDATES:]
    image_budget = _MAX_IMAGES_PER_EPISODE
    images: list[dict[str, Any]] = []

    coarse_grids = _coarse_grids(record)
    if not coarse_grids:
        coarse_grids = [uniform_indices(episode.length, manifest.fps, config.sampling.coarse_fps,
                                        config.sampling.coarse_max_frames)]
    for pass_id, grid in enumerate(coarse_grids[:2]):
        if image_budget <= 0:
            break
        indices = grid[:image_budget]
        samples = _sample(service, episode, config.primary_camera, indices, manifest.fps)
        for sample in samples:
            name = f"coarse-{pass_id}-frame-{sample.frame_index}.jpg"
            _write_file(assets / name, sample.jpeg)
            images.append(_image_metadata(episode_name, name, sample, "coarse"))
        image_budget -= len(samples)

    camera_order = list(dict.fromkeys([config.primary_camera, *config.refine_cameras]))
    radius = max(0, round(config.sampling.refine_window_seconds * manifest.fps))
    stride = max(1, round(manifest.fps / config.sampling.refine_fps))
    candidates: list[int] = []
    for position, candidate in enumerate(candidate_queue):
        context = window_indices(candidate, radius, stride, episode.length)
        context = sorted(set(context) | {max(0, candidate - 1), candidate})
        required = len(context) * len(camera_order)
        if required > image_budget:
            omitted = candidate_queue[position:] + omitted
            break
        candidates.append(candidate)
        for camera in camera_order:
            requested = context[:image_budget]
            samples = _sample(service, episode, camera, requested, manifest.fps)
            slug = _camera_slug(camera)
            by_index = {}
            for sample in samples:
                name = f"boundary-{candidate}-{slug}-frame-{sample.frame_index}.jpg"
                _write_file(assets / name, sample.jpeg)
                images.append(_image_metadata(episode_name, name, sample, "boundary_context"))
                by_index[sample.frame_index] = sample
            if camera == config.primary_camera:
                before = max(0, candidate - 1)
                after = candidate
                _write_file(assets / f"boundary-{candidate}-before.jpg", by_index[before].jpeg)
                _write_file(assets / f"boundary-{candidate}-after.jpg", by_index[after].jpeg)
            image_budget -= len(samples)

    payload = {
        "episode": episode_name,
        "episode_index": record.episode_index,
        "source_fingerprint": record.source_fingerprint,
        "review_reasons": list(record.review_reasons),
        "validation_issues": [item.model_dump(mode="json") for item in record.validation_issues],
        "candidate_annotation": _candidate_annotation(record),
        "candidates": candidates,
        "omitted_candidates": omitted,
        "candidate_cap": _MAX_CANDIDATES,
        "image_cap": _MAX_IMAGES_PER_EPISODE,
        "coarse_attempts": [item.model_dump(mode="json") for item in record.coarse_attempts],
        "refine_attempts": [item.model_dump(mode="json") for item in record.refine_attempts],
        "decision_summaries": _decision_summaries(record),
        "images": images,
    }
    _write_file(staging / f"{episode_name}.json", _canonical_json(payload).encode("utf-8"))
    return payload


def _sample(service, episode, camera, indices, fps):
    if camera not in episode.videos:
        raise ValueError(f"review camera {camera!r} is absent from episode")
    if indices != sorted(set(indices)) or any(type(item) is not int or not 0 <= item < episode.length for item in indices):
        raise ValueError("review sampling indices are invalid")
    samples = service.sampler(episode.videos[camera], camera, indices, fps)
    if not isinstance(samples, list) or len(samples) != len(indices):
        raise ValueError("review sampler must return exactly the requested frames")
    if [item.frame_index for item in samples] != indices:
        raise ValueError("review sampler frame order does not match request")
    for item in samples:
        if not isinstance(item, FrameSample) or item.camera_key != camera:
            raise ValueError("review sampler camera metadata is invalid")
        if not math.isclose(item.timestamp_seconds, item.frame_index / fps, abs_tol=1e-7):
            raise ValueError("review sampler timestamp is invalid")
    return samples


def _candidate_frames(record, frame_count):
    found: set[int] = set()
    for attempt in record.coarse_attempts:
        found.update(item.estimated_frame for item in attempt.coarse_boundaries)
    found.update(item.boundary_frame for item in record.refine_attempts)
    candidate = _candidate_annotation(record)
    if candidate:
        found.update(candidate["boundaries"])
    return sorted(item for item in found if type(item) is int and 0 <= item < frame_count)


def _candidate_annotation(record):
    if record.final_annotation is not None:
        return record.final_annotation.model_dump(mode="json")
    for key in ("refine_decision", "coarse_decision"):
        value = record.sampling_details.get(key)
        if isinstance(value, dict):
            candidate = value.get("candidate_annotation")
            if isinstance(candidate, dict):
                start = candidate.get("start_subtask_index")
                boundaries = candidate.get("boundaries")
                if type(start) is int and isinstance(boundaries, list) and all(type(item) is int for item in boundaries):
                    return {"start_subtask_index": start, "boundaries": list(boundaries)}
    if record.coarse_attempts:
        first = record.coarse_attempts[0]
        return {"start_subtask_index": first.start_subtask_index,
                "boundaries": [item.estimated_frame for item in first.coarse_boundaries]}
    return None


def _coarse_grids(record):
    decision = record.sampling_details.get("coarse_decision")
    if not isinstance(decision, dict):
        return []
    raw = decision.get("sampled_frame_indices")
    if not isinstance(raw, list):
        return []
    result = []
    for grid in raw:
        if isinstance(grid, list) and all(type(item) is int for item in grid):
            result.append(grid)
    return result


def _decision_summaries(record):
    allowed = {
        "coarse_decision": {"status", "reasons", "start_subtask_index", "observed_subtask_indices",
                            "boundary_centers", "sampled_frame_indices"},
        "refine_decision": {"status", "reasons", "start_subtask_index", "observed_subtask_indices",
                            "coarse_boundary_centers", "candidate_annotation", "failure_category",
                            "camera_order", "provenance"},
    }
    summaries = {}
    for key, fields in allowed.items():
        value = record.sampling_details.get(key)
        if isinstance(value, dict):
            summary = {field: value[field] for field in fields
                       if field in value and field not in {"provenance", "candidate_annotation"}}
            if key == "refine_decision":
                candidate = _candidate_annotation(record)
                if candidate is not None:
                    summary["candidate_annotation"] = candidate
            provenance = value.get("provenance")
            if key == "refine_decision" and isinstance(provenance, list):
                safe_fields = {"boundary_index", "from_subtask_index", "to_subtask_index", "stage",
                               "pass_id", "request_center", "radius_frames", "stride", "cameras",
                               "samples", "outcome"}
                safe_provenance = []
                for item in provenance:
                    if not isinstance(item, dict):
                        continue
                    safe_item = {field: item[field] for field in safe_fields
                                 if field in item and field != "samples"}
                    samples = item.get("samples")
                    if isinstance(samples, list):
                        safe_item["samples"] = [
                            {field: sample[field] for field in {"camera_key", "frame_indices"}
                             if field in sample}
                            for sample in samples if isinstance(sample, dict)
                        ]
                    safe_provenance.append(safe_item)
                summary["provenance"] = safe_provenance
            summaries[key] = summary
    return summaries


def _image_metadata(episode, name, sample, purpose):
    return {"path": f"{episode}/{name}", "camera": sample.camera_key,
            "frame_index": sample.frame_index, "timestamp_seconds": sample.timestamp_seconds,
            "purpose": purpose}


def _effective_config(manifest: RunManifest, root: Path) -> AnnotationConfig:
    raw = json.loads(_canonical_json(manifest.effective_config))
    if not isinstance(raw, dict):
        raise ValueError("manifest effective_config must be an object")
    raw["source"] = str(manifest.dataset_root)
    raw["work_dir"] = str(root)
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("manifest effective_config model is invalid")
    model["api_key"] = "review-local-redacted"
    config = AnnotationConfig.model_validate_json(_canonical_json(raw), strict=True)
    if (config.mode != manifest.mode or config.high_level_instruction != manifest.high_level_instruction
            or config.subtasks != manifest.subtasks
            or config.sampling.min_segment_frames != manifest.min_segment_frames):
        raise ValueError("manifest effective configuration is inconsistent")
    configured_cameras = [config.primary_camera, *config.refine_cameras]
    if any(camera not in manifest.camera_keys for camera in configured_cameras):
        raise ValueError("manifest effective configuration references an unavailable camera")
    return config


def _validate_dataset(manifest, dataset):
    if not isinstance(dataset, DatasetIndex):
        raise TypeError("inspect_dataset must return DatasetIndex")
    if (dataset.root.resolve() != manifest.dataset_root.resolve() or dataset.version != manifest.dataset_version
            or not math.isclose(dataset.fps, manifest.fps, abs_tol=1e-9)
            or list(dataset.camera_keys) != list(manifest.camera_keys)
            or len(dataset.episodes) != manifest.total_episodes
            or [item.length for item in dataset.episodes] != list(manifest.episode_lengths)
            or sum(item.length for item in dataset.episodes) != manifest.total_frames
            or [item.episode_index for item in dataset.episodes] != list(range(manifest.total_episodes))):
        raise ValueError("current dataset inspection is incompatible with workspace manifest")


def _assert_current_source(manifest, dataset, record):
    current = compute_source_fingerprint(manifest.dataset_root, dataset.episodes[record.episode_index])
    if current != record.source_fingerprint:
        raise ValueError(f"episode {record.episode_index} source fingerprint is stale")


def _load_manifest(root):
    payload = _strict_json(_read_regular(root / "manifest.json", _MAX_MANIFEST_BYTES))
    return RunManifest.model_validate_json(_canonical_json(payload))


def _workspace_root(path):
    if not isinstance(path, Path):
        raise TypeError("work_dir must be a Path")
    if path.is_symlink():
        raise ValueError("workspace root must not be a symlink")
    root = path.resolve()
    if not root.is_dir():
        raise ValueError("workspace root must be a directory")
    return root


def _read_regular(path, limit):
    if path.is_symlink():
        raise ValueError("input must not be a symlink")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("input must be a bounded regular file")
        chunks, remaining = [], info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("input changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(payload, object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))
    if not isinstance(value, dict):
        raise ValueError("JSON must contain an object")
    return value


def _assert_directory(path, label):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory")


def _assert_replaceable_destination(path):
    if path.is_symlink():
        raise ValueError("review destination must not be a symlink")
    if not path.is_dir():
        raise ValueError("review destination must be a directory")
    entries = list(path.iterdir())
    if entries and not (path / _OWNER).is_file():
        raise ValueError("nonempty review destination is not owned by this renderer")
    allowed_root = {_OWNER, "review.js", "index.html"}
    for item in entries:
        if item.is_symlink():
            raise ValueError("review destination must not contain symlinks")
        if item.name in allowed_root:
            if not item.is_file():
                raise ValueError("review destination contains an invalid generated entry")
            continue
        if re.fullmatch(r"episode_[0-9]{6}\.json", item.name):
            if not item.is_file():
                raise ValueError("review JSON output must be a regular file")
            continue
        if re.fullmatch(r"episode_[0-9]{6}", item.name):
            if not item.is_dir():
                raise ValueError("review asset output must be a directory")
            for asset in item.iterdir():
                if asset.is_symlink() or not asset.is_file() or not asset.name.endswith(".jpg"):
                    raise ValueError("review asset directory contains an unexpected entry")
            continue
        raise ValueError("review destination contains an unowned entry")


def _publish_directory(staging, destination):
    backup = destination.parent / f".needs_review.backup-{secrets.token_hex(12)}"
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


def _write_file(path, payload):
    if path.exists() or path.is_symlink():
        raise ValueError("review output collision")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("review output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _camera_slug(camera):
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", camera).strip("-")[:40] or "camera"
    return f"{readable}-{hashlib.sha256(camera.encode()).hexdigest()[:8]}"


def _strict_time(value, prior):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("review clock must return a timezone-aware datetime")
    return max(value.astimezone(UTC), prior + timedelta(microseconds=1))


def _append_human_outbox(details, record, updated_at):
    key = "_pipeline_transition_events"
    if key not in details:
        return
    events = details[key]
    if not isinstance(events, list):
        raise ValueError("pipeline transition outbox is invalid")
    payload = {"run_fingerprint": record.run_fingerprint, "episode": record.episode_index,
               "from_status": "needs_review", "to_status": "accepted",
               "updated_at": updated_at.isoformat(), "event": "accepted",
               "category": None, "reasons": []}
    event = {"event_id": hashlib.sha256(_canonical_json(payload).encode()).hexdigest(),
             "timestamp": updated_at.isoformat(), "episode": record.episode_index,
             "from_status": "needs_review", "to_status": "accepted", "event": "accepted",
             "category": None, "reasons": []}
    details[key] = [*events, event]
