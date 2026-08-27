"""Builders for small, structurally valid LeRobot v2.1 datasets."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robo_annotate.config import AnnotationConfig


def make_legacy_v4_workspace(tmp_path: Path) -> Path:
    """Write one realistic coarse-v4 episode record with the removed field."""
    work = tmp_path / "legacy-v4-work"
    episodes = work / "episodes"
    episodes.mkdir(parents=True)
    for relative in ("previews", "previews/needs_review", "logs"):
        (work / relative).mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 8, 22, tzinfo=UTC).isoformat()
    payload = {
        "episode_index": 0,
        "status": "coarse_done",
        "coarse_attempts": [{
            "start_subtask_index": 0,
            "observed_subtask_indices": [0, 1],
            "coarse_boundaries": [{
                "from_subtask_index": 0,
                "to_subtask_index": 1,
                "estimated_frame": 20,
                "evidence": "visible transition",
            }],
            "confidence": 0.9,
            "uncertainties": [],
        }],
        "refine_attempts": [],
        "final_annotation": None,
        "validation_issues": [],
        "review_reasons": [],
        "failure_category": None,
        "decision_source": None,
        "created_at": now,
        "updated_at": now,
        "source_fingerprint": "a" * 64,
        "run_fingerprint": "b" * 64,
        "prompt_version": "coarse-v4/refine-v1",
        "model_revision": "c" * 40,
        "sampling_details": {"coarse_decision": {"legacy": True}},
    }
    (episodes / "episode_000000.json").write_text(json.dumps(payload), encoding="utf-8")
    return work


def make_lerobot_fixture(
    tmp_path: Path,
    lengths: list[int],
    fps: float,
    cameras: list[str],
) -> Path:
    """Create a minimal dataset whose video paths can be probed by a fake."""
    root = tmp_path / "source"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "chunks_size": 2,
        "total_chunks": (len(lengths) + 1) // 2,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {camera: {"dtype": "video"} for camera in cameras},
        "total_videos": len(lengths) * len(cameras),
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "arrange items"}) + "\n",
        encoding="utf-8",
    )
    episode_lines: list[str] = []
    for index, length in enumerate(lengths):
        episode_lines.append(
            json.dumps(
                {"episode_index": index, "tasks": ["arrange items"], "length": length}
            )
        )
        chunk = index // 2
        parquet = root / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame": list(range(length))}), parquet)
        for camera in cameras:
            video = root / f"videos/chunk-{chunk:03d}/{camera}/episode_{index:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()
    (meta / "episodes.jsonl").write_text("\n".join(episode_lines) + "\n", encoding="utf-8")
    return root


def make_config(
    root: Path,
    work: Path,
    primary: str = "cam.eye",
    refine: list[str] | None = None,
    mode: str = "complete",
) -> AnnotationConfig:
    return AnnotationConfig.model_validate(
        {
            "source": root,
            "work_dir": work,
            "mode": mode,
            "high_level_instruction": "Arrange the items.",
            "primary_camera": primary,
            "refine_cameras": refine or [primary],
            "subtasks": [{"skill": "pick", "text": "Pick an item."}],
        }
    )
