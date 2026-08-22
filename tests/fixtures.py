"""Builders for small, structurally valid LeRobot v2.1 datasets."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from qwen_annotate.config import AnnotationConfig


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
