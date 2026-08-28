import errno
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from robo_annotate import v30_video_writer, writer_publication
from robo_annotate.lerobot import inspect_dataset
from robo_annotate.v30_video_writer import write_v30_video_subset
from tests.v30_fixtures import (
    colors_for_episode,
    decoded_colors,
    make_lerobot_v30_fixture,
    make_v30_config,
    read_v30_info,
    source_tree_digest,
)


def test_reencodes_selected_video_slices_without_middle_episode(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    before_source = source_tree_digest(root)

    result = write_v30_video_subset(
        tmp_path / "staging",
        dataset,
        [0, 2],
        read_v30_info(root),
    )

    placements = result.placements["observation.images.main"]
    assert [(item.from_timestamp, item.to_timestamp) for item in placements] == [
        (pytest.approx(0.0), pytest.approx(1.2)),
        (pytest.approx(1.2), pytest.approx(2.2)),
    ]
    assert decoded_colors(
        result.files_by_camera["observation.images.main"][0]
    ) == (
        colors_for_episode(0, 0, 6)
        + colors_for_episode(0, 2, 5)
    )
    assert source_tree_digest(root) == before_source


def test_accepts_one_frame_timestamp_tolerance_without_writing_neighbor(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    first = dataset.episodes[0]
    videos = {
        camera: reference.model_copy(
            update={
                "to_timestamp": reference.to_timestamp
                + 0.4 / dataset.fps,
            }
        )
        for camera, reference in first.videos.items()
    }
    tolerant_dataset = dataset.model_copy(
        update={
            "episodes": [
                first.model_copy(update={"videos": videos}),
                *dataset.episodes[1:],
            ]
        }
    )

    result = write_v30_video_subset(
        tmp_path / "staging",
        tolerant_dataset,
        [0],
        read_v30_info(root),
    )

    placement = result.placements["observation.images.main"][0]
    assert placement.from_timestamp == pytest.approx(0.0)
    assert placement.to_timestamp == pytest.approx(1.2)
    assert decoded_colors(
        result.files_by_camera["observation.images.main"][0]
    ) == colors_for_episode(0, 0, 6)


def test_rebuilds_two_cameras_in_selected_order_with_exact_video_facts(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_video_subset(
        tmp_path / "staging",
        dataset,
        [2, 0],
        read_v30_info(root),
    )

    assert set(result.placements) == set(dataset.camera_keys)
    for camera_index, camera in enumerate(dataset.camera_keys):
        placements = result.placements[camera]
        assert [item.source_index for item in placements] == [2, 0]
        assert [item.output_index for item in placements] == [0, 1]
        assert [(item.chunk_index, item.file_index) for item in placements] == [
            (0, 0),
            (0, 0),
        ]
        assert [(item.from_timestamp, item.to_timestamp) for item in placements] == [
            (pytest.approx(0.0), pytest.approx(1.0)),
            (pytest.approx(1.0), pytest.approx(2.2)),
        ]
        output = result.files_by_camera[camera][0]
        assert decoded_colors(output) == (
            colors_for_episode(camera_index, 2, 5)
            + colors_for_episode(camera_index, 0, 6)
        )
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            assert float(stream.average_rate) == pytest.approx(5.0)
            assert (stream.width, stream.height) == (32, 24)
            assert sum(1 for _ in container.decode(stream)) == 11


def test_packs_whole_episodes_across_numeric_video_chunks(
    tmp_path: Path,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    info = read_v30_info(root)
    info["chunks_size"] = 2
    info["video_files_size_in_mb"] = 15_000 / (1024 * 1024)

    result = write_v30_video_subset(
        tmp_path / "staging",
        dataset,
        [0, 1, 2],
        info,
    )

    for camera_index, camera in enumerate(dataset.camera_keys):
        assert [
            (item.chunk_index, item.file_index)
            for item in result.placements[camera]
        ] == [(0, 0), (0, 1), (1, 0)]
        assert [
            path.relative_to(tmp_path / "staging").as_posix()
            for path in result.files_by_camera[camera]
        ] == [
            f"videos/{camera}/chunk-000/file-000.mp4",
            f"videos/{camera}/chunk-000/file-001.mp4",
            f"videos/{camera}/chunk-001/file-000.mp4",
        ]
        assert [decoded_colors(path) for path in result.files_by_camera[camera]] == [
            colors_for_episode(camera_index, 0, 6),
            colors_for_episode(camera_index, 1, 8),
            colors_for_episode(camera_index, 2, 5),
        ]


def test_preserves_non_padded_video_path_template(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(
        tmp_path,
        non_padded_video_template=True,
    )
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))

    result = write_v30_video_subset(
        tmp_path / "staging",
        dataset,
        [0],
        read_v30_info(root),
    )

    assert result.files_by_camera["observation.images.main"][0].name == "file-0.mp4"


def test_rejects_preexisting_videos_without_clobbering_it(tmp_path: Path) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    (staging / "videos").mkdir(parents=True)
    competitor = staging / "videos/competitor.txt"
    competitor.write_bytes(b"preserve competitor")
    before_source = source_tree_digest(root)

    with pytest.raises(ValueError, match="unsafe staging videos entry already exists"):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert competitor.read_bytes() == b"preserve competitor"
    assert source_tree_digest(root) == before_source
    assert list(staging.glob(".v30-video-*")) == []


def test_decode_failure_rolls_back_every_owned_video_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)

    def fail_decode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected video decode failure")

    monkeypatch.setattr(v30_video_writer, "_decode_episode_frames", fail_decode)

    with pytest.raises(RuntimeError, match="injected video decode failure") as raised:
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert not any(
        "writer rollback left owned filesystem identities active" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert source_tree_digest(root) == before_source
    assert not staging.exists()


def test_source_path_replacement_is_detected_after_stable_handle_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    source_video = dataset.episodes[0].videos[
        "observation.images.main"
    ].path
    displaced = tmp_path / "displaced-source.mp4"
    actual_open = v30_video_writer.av.open
    injected = False

    def open_then_replace_source(path: object, *args: object, **kwargs: object):
        nonlocal injected
        container = actual_open(path, *args, **kwargs)
        if (
            not injected
            and str(kwargs.get("mode", args[0] if args else "r")) == "r"
            and isinstance(path, str)
            and path.startswith("/proc/self/fd/")
        ):
            try:
                opened_path = Path(os.readlink(path))
            except OSError:
                opened_path = Path()
            if opened_path == source_video:
                os.replace(source_video, displaced)
                source_video.write_bytes(b"preserve replacement")
                injected = True
        return container

    monkeypatch.setattr(v30_video_writer.av, "open", open_then_replace_source)

    with pytest.raises(
        ValueError,
        match="video shard changed during validation|source changed during video publication",
    ):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert displaced.is_file()
    assert source_video.read_bytes() == b"preserve replacement"
    assert not (staging / "videos").exists()


def test_final_staging_close_failure_rolls_back_published_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_close = writer_publication._AnchoredDirectoryPath.close
    injected = False

    def close_then_fail_final_staging(anchor: object) -> None:
        nonlocal injected
        actual_close(anchor)
        if getattr(anchor, "label") == "staging path" and not injected:
            injected = True
            raise OSError(errno.EIO, "injected final staging anchor close failure")

    monkeypatch.setattr(
        writer_publication._AnchoredDirectoryPath,
        "close",
        close_then_fail_final_staging,
    )

    with pytest.raises(ValueError, match="cleanup failed"):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert not (staging / "videos").exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_publish_race_preserves_competitor_and_removes_owned_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    displaced = staging / "displaced-videos"
    before_source = source_tree_digest(root)
    actual_publish = writer_publication.rename_noreplace_at
    injected = False

    def publish_then_replace_videos(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        actual_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        if source_name == "videos" and destination_name == "videos" and not injected:
            os.replace(staging / "videos", displaced)
            (staging / "videos").mkdir()
            (staging / "videos/competitor.txt").write_bytes(b"preserve competitor")
            injected = True

    monkeypatch.setattr(
        writer_publication,
        "rename_noreplace_at",
        publish_then_replace_videos,
    )

    with pytest.raises(ValueError, match="staging videos changed during publication"):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert (staging / "videos/competitor.txt").read_bytes() == b"preserve competitor"
    assert not displaced.exists()
    assert list(staging.glob(".v30-video-*")) == []
    assert source_tree_digest(root) == before_source


def test_encoder_close_failure_rolls_back_staged_and_published_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    actual_open = v30_video_writer.av.open
    injected = False

    class CloseFailure:
        def __init__(self, container: object) -> None:
            self.container = container

        def __getattr__(self, name: str):
            return getattr(self.container, name)

        def close(self) -> None:
            nonlocal injected
            self.container.close()
            injected = True
            raise OSError(errno.EIO, "injected encoder close failure")

    def wrap_output(path: object, *args: object, **kwargs: object):
        container = actual_open(path, *args, **kwargs)
        mode = str(kwargs.get("mode", args[0] if args else "r"))
        if mode == "w" and not injected:
            return CloseFailure(container)
        return container

    monkeypatch.setattr(v30_video_writer.av, "open", wrap_output)

    with pytest.raises(ValueError, match="video encoding cleanup failed"):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert not staging.exists()


def test_replaced_owned_video_file_preserves_competitor_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    actual_open = v30_video_writer.av.open
    displaced: Path | None = None
    competitor: Path | None = None

    def open_then_replace_output(path: object, *args: object, **kwargs: object):
        nonlocal displaced, competitor
        container = actual_open(path, *args, **kwargs)
        mode = str(kwargs.get("mode", args[0] if args else "r"))
        if mode == "w" and displaced is None:
            target = Path(os.readlink(str(path)))
            displaced = target.with_name("displaced-owned-video.mp4")
            os.replace(target, displaced)
            target.write_bytes(b"preserve competitor")
            competitor = target
        return container

    monkeypatch.setattr(v30_video_writer.av, "open", open_then_replace_output)

    with pytest.raises(ValueError, match="changed during video encoding"):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert displaced is not None
    assert competitor is not None
    assert competitor.read_bytes() == b"preserve competitor"
    assert not displaced.exists()
    assert not (staging / "videos").exists()
    assert source_tree_digest(root) == before_source


def test_owned_video_descriptor_close_failure_rolls_back_published_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    before_source = source_tree_digest(root)
    before_fds = len(os.listdir("/proc/self/fd"))
    actual_close = writer_publication.OwnedFile.close
    injected = False

    def close_then_fail_once(output: writer_publication.OwnedFile) -> None:
        nonlocal injected
        actual_close(output)
        if not injected:
            injected = True
            raise OSError(errno.EIO, "injected owned video descriptor close failure")

    monkeypatch.setattr(
        writer_publication.OwnedFile,
        "close",
        close_then_fail_once,
    )

    with pytest.raises(ValueError, match="cleanup failed"):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert injected
    assert source_tree_digest(root) == before_source
    assert not (staging / "videos").exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds


class _SourceFrame:
    def __init__(self, pts: int | float | None, shape: tuple[int, int]) -> None:
        self.pts = pts
        self._shape = shape

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        width, height = self._shape
        return np.zeros((height, width, 3), dtype=np.uint8)


class _SourceContainer:
    def __init__(
        self,
        pts: list[int | float | None],
        *,
        fps: Fraction = Fraction(5, 1),
        time_base: Fraction | int = Fraction(1, 5),
        width: int = 32,
        height: int = 24,
        close_error: BaseException | None = None,
    ) -> None:
        self.streams = [
            SimpleNamespace(
                type="video",
                average_rate=fps,
                time_base=time_base,
                width=width,
                height=height,
            )
        ]
        self._frames = [_SourceFrame(value, (width, height)) for value in pts]
        self._close_error = close_error
        self.closed = False

    def seek(self, offset: int, **kwargs: object) -> None:
        del offset, kwargs

    def decode(self, stream: object):
        del stream
        yield from self._frames

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


@pytest.mark.parametrize(
    ("container", "message"),
    [
        (_SourceContainer(list(range(6)), fps=Fraction(4, 1)), "fps"),
        (_SourceContainer(list(range(6)), time_base=0), "time base"),
        (_SourceContainer(list(range(6)), width=16), "dimensions"),
        (_SourceContainer([None, *range(1, 6)]), "without PTS"),
        (_SourceContainer([0, 1, 1, 3, 4, 5]), "duplicate"),
        (_SourceContainer(list(range(5))), "missing episode frame"),
    ],
)
def test_rejects_invalid_source_video_facts_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container: _SourceContainer,
    message: str,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    staging = tmp_path / "staging"
    _replace_first_source_open(monkeypatch, container)

    with pytest.raises(ValueError, match=message):
        write_v30_video_subset(
            staging,
            dataset,
            [0],
            read_v30_info(root),
        )

    assert container.closed
    assert not staging.exists()


def test_source_close_failure_does_not_mask_primary_decode_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    container = _SourceContainer(
        list(range(6)),
        fps=Fraction(4, 1),
        close_error=OSError(errno.EIO, "secondary source close failure"),
    )
    _replace_first_source_open(monkeypatch, container)

    with pytest.raises(ValueError, match="fps") as raised:
        write_v30_video_subset(
            tmp_path / "staging",
            dataset,
            [0],
            read_v30_info(root),
        )

    assert any(
        "secondary source close failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )


def test_does_not_round_preceding_slice_pts_into_local_frame_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_lerobot_v30_fixture(tmp_path)
    dataset = inspect_dataset(make_v30_config(root, tmp_path / "work"))
    container = _SourceContainer([5.5, *range(6, 14)])
    _replace_first_source_open(monkeypatch, container)

    result = write_v30_video_subset(
        tmp_path / "staging",
        dataset,
        [1],
        read_v30_info(root),
    )

    output = result.files_by_camera["observation.images.main"][0]
    with av.open(str(output)) as decoded:
        assert sum(1 for _ in decoded.decode(video=0)) == 8


def _replace_first_source_open(
    monkeypatch: pytest.MonkeyPatch,
    replacement: _SourceContainer,
) -> None:
    actual_open = v30_video_writer.av.open
    replaced = False

    def route_open(path: object, *args: object, **kwargs: object):
        nonlocal replaced
        mode = str(kwargs.get("mode", args[0] if args else "r"))
        if mode == "r" and not replaced:
            replaced = True
            return replacement
        return actual_open(path, *args, **kwargs)

    monkeypatch.setattr(v30_video_writer.av, "open", route_open)
