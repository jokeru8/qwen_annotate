import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.v30_fixtures import make_lerobot_v30_fixture


def test_v30_fixture_uses_shared_real_payloads(tmp_path: Path) -> None:
    """A v3 fixture must represent real shared Parquet and MP4 payloads."""
    root = make_lerobot_v30_fixture(tmp_path)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")

    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert episodes.num_rows == 3
    assert len(list((root / "data").glob("**/*.parquet"))) == 1
    assert len(list((root / "videos/observation.images.main").glob("**/*.mp4"))) == 1
    with av.open(str(next((root / "videos/observation.images.main").glob("**/*.mp4")))) as container:
        assert sum(1 for _ in container.decode(video=0)) == 19


def test_v30_fixture_preserves_declared_array_dimensions_in_stats(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet")

    np.testing.assert_allclose(
        stats["observation.matrix"]["mean"],
        [[18 / 19, 53 / 19], [71 / 19, 35 / 19]],
    )
    np.testing.assert_allclose(
        episodes["stats/observation.matrix/mean"].to_pylist()[0],
        [[0.0, 2.5], [2.5, 2.5]],
    )
    assert episodes.schema.field("stats/observation.matrix/mean").type.equals(
        pa.list_(pa.list_(pa.float64()))
    )


def test_v30_fixture_rejects_lengths_beyond_color_capacity_before_writes(tmp_path: Path) -> None:
    """The color-coded fixture must reject unsupported episode lengths atomically."""
    with pytest.raises(ValueError, match="lengths"):
        make_lerobot_v30_fixture(tmp_path, lengths=(13,))

    assert not (tmp_path / "lerobot-v30").exists()
