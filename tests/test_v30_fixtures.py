import json
from pathlib import Path

import av
import pyarrow.parquet as pq

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
