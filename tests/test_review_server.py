from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from robo_annotate.lerobot import inspect_dataset
from robo_annotate.review_server import create_review_app
from robo_annotate.workspace import WorkspaceStore
from tests.test_review import _workspace
from tests.v30_fixtures import make_lerobot_v30_fixture, make_v30_config


def _client(tmp_path: Path):
    work, _, store, record, services, _ = _workspace(tmp_path)
    return TestClient(create_review_app(work, services=services)), work, store, record, services


def v30_review_client(tmp_path: Path, episode_index: int) -> TestClient:
    root = make_lerobot_v30_fixture(tmp_path)
    config = make_v30_config(root, tmp_path / "work-v30")
    dataset = inspect_dataset(config)
    store = WorkspaceStore(config.work_dir)
    store.initialize(config, dataset, "a" * 40)
    store.load_episode(episode_index)
    return TestClient(create_review_app(config.work_dir))


def test_session_and_episode_views_are_bounded_and_redacted(tmp_path: Path) -> None:
    client, _, _, record, _ = _client(tmp_path)

    session = client.get("/api/session")
    assert session.status_code == 200
    assert session.json() == {
        "mode": "dagger_patch",
        "fps": 20.0,
        "camera_keys": ["cam.eye", "cam/wrist"],
        "primary_camera": "cam.eye",
        "high_level_instruction": "Arrange <script>alert(1)</script>",
        "subtasks": [
            {"skill": "pick", "text": "Pick <img src=x onerror=alert(2)>"},
            {"skill": "place", "text": "Place"},
            {"skill": "finish", "text": "Finish"},
        ],
        "total_episodes": 1,
        "min_segment_frames": 8,
        "status_counts": {"needs_review": 1},
    }
    listing = client.get("/api/episodes", params={"status": "needs_review"})
    assert listing.status_code == 200
    assert listing.json()[0]["episode_index"] == 0
    assert listing.json()[0]["status"] == "needs_review"
    assert client.get("/api/episodes", params={"status": "failed"}).json() == []

    detail = client.get("/api/episodes/0")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["source_fingerprint"] == record.source_fingerprint
    assert payload["candidate_annotation"] == {"start_subtask_index": 1, "boundaries": [184]}
    assert payload["episode_length"] == 240
    assert payload["videos"]["cam.eye"] == {
        "url": "/api/episodes/0/videos/cam.eye",
        "from_timestamp": 0.0,
        "to_timestamp": 12.0,
    }
    serialized = detail.text
    assert "TOP-SECRET" not in serialized and "NESTED-SECRET" not in serialized
    assert "sampling_details" not in serialized


def test_invalid_status_episode_and_camera_fail_without_path_leaks(tmp_path: Path) -> None:
    client, work, _, _, _ = _client(tmp_path)
    assert client.get("/api/episodes", params={"status": "unknown"}).status_code == 422
    assert client.get("/api/episodes/99").status_code == 404
    traversal = quote("../manifest.json", safe="")
    response = client.get(f"/api/episodes/0/videos/{traversal}")
    assert response.status_code == 404
    assert str(work) not in response.text and "manifest.json" not in response.text


def test_video_endpoint_supports_full_and_single_ranges(tmp_path: Path) -> None:
    client, _, _, _, _ = _client(tmp_path)
    url = "/api/episodes/0/videos/cam.eye"
    full = client.get(url)
    assert full.status_code == 200 and full.content == b"video"
    assert full.headers["accept-ranges"] == "bytes"

    partial = client.get(url, headers={"Range": "bytes=1-3"})
    assert partial.status_code == 206 and partial.content == b"ide"
    assert partial.headers["content-range"] == "bytes 1-3/5"
    suffix = client.get(url, headers={"Range": "bytes=-2"})
    assert suffix.status_code == 206 and suffix.content == b"eo"
    open_ended = client.get(url, headers={"Range": "bytes=3-"})
    assert open_ended.status_code == 206 and open_ended.content == b"eo"
    slash_camera_url = client.get("/api/episodes/0").json()["videos"]["cam/wrist"]["url"]
    slash_camera = client.get(slash_camera_url)
    assert slash_camera.status_code == 200 and slash_camera.content == b"video"


def test_v30_episode_payload_exposes_shared_video_offsets(tmp_path: Path) -> None:
    client = v30_review_client(tmp_path, episode_index=1)

    source = client.get("/api/episodes/1").json()["videos"]["observation.images.main"]

    assert source["url"].endswith("observation.images.main")
    assert source["from_timestamp"] == pytest.approx(1.2)
    assert source["to_timestamp"] == pytest.approx(2.8)


def test_video_endpoint_rejects_multiple_or_unsatisfiable_ranges(tmp_path: Path) -> None:
    client, _, _, _, _ = _client(tmp_path)
    url = "/api/episodes/0/videos/cam.eye"
    for value in ("items=0-1", "bytes=0-1,3-4", "bytes=9-", "bytes=3-1", "bytes=-0"):
        response = client.get(url, headers={"Range": value})
        assert response.status_code == 416
        assert response.headers["content-range"] == "bytes */5"


def _decision_payload(record, *, takeover=False, boundaries=(184,)):
    return {
        "episode_index": record.episode_index,
        "source_fingerprint": record.source_fingerprint,
        "run_fingerprint": record.run_fingerprint,
        "mode": "dagger_patch",
        "expected_status": record.status,
        "expected_updated_at": record.updated_at.isoformat(),
        "takeover_confirmed": takeover,
        "start_subtask_index": 1,
        "boundaries": list(boundaries),
        "note": "reviewed in browser",
    }


def test_decision_endpoint_accepts_needs_review_candidate(tmp_path: Path) -> None:
    client, _, store, record, _ = _client(tmp_path)
    response = client.post("/api/episodes/0/decision", json=_decision_payload(record))
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert response.json()["decision_source"] == "human"
    assert store.load_episode(0).final_annotation.boundaries == [184]


def test_decision_endpoint_requires_takeover_for_pending_and_reports_conflict(tmp_path: Path) -> None:
    client, _, store, _, services = _client(tmp_path)
    pending = store.invalidate_episode(0, episode=services.inspect_dataset(None).episodes[0])
    response = client.post("/api/episodes/0/decision", json=_decision_payload(pending))
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "takeover_required"

    payload = _decision_payload(pending, takeover=True)
    payload["expected_updated_at"] = "2020-01-01T00:00:00Z"
    stale = client.post("/api/episodes/0/decision", json=payload)
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "stale_episode"


def test_decision_endpoint_maps_invalid_annotation_and_identity_safely(tmp_path: Path) -> None:
    client, work, _, record, _ = _client(tmp_path)
    invalid = client.post(
        "/api/episodes/0/decision", json=_decision_payload(record, boundaries=(5,)),
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "invalid_annotation"
    assert invalid.json()["detail"]["issues"] == ["segment_too_short"]

    payload = _decision_payload(record)
    payload["source_fingerprint"] = "f" * 64
    conflict = client.post("/api/episodes/0/decision", json=payload)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "workspace_conflict"
    assert str(work) not in conflict.text

    payload = _decision_payload(record)
    payload["episode_index"] = 1
    mismatch = client.post("/api/episodes/0/decision", json=payload)
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["code"] == "episode_mismatch"


def test_decision_endpoint_rejects_oversized_body_before_validation(tmp_path: Path) -> None:
    client, _, _, _, _ = _client(tmp_path)
    response = client.post(
        "/api/episodes/0/decision",
        content=b"{" + b" " * (129 * 1024) + b"}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_open_video_descriptor_keeps_verified_bytes_after_path_replacement(tmp_path: Path) -> None:
    client, _, _, _, _ = _client(tmp_path)
    runtime = client.app.state.review_runtime
    descriptor, size = runtime.open_video(0, "cam.eye")
    path = runtime.dataset.episodes[0].videos["cam.eye"].path
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(b"other")
    replacement.replace(path)
    try:
        assert b"".join(__import__("robo_annotate.review_server", fromlist=["_fd_chunks"])._fd_chunks(descriptor, 0, size)) == b"video"
    finally:
        try:
            __import__("os").close(descriptor)
        except OSError:
            pass
