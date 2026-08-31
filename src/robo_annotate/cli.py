"""Operational command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from .config import load_config
from .converter import convert_dataset
from .evaluation import evaluate_complete, write_evaluation_report
from .lerobot import inspect_dataset
from .model_manager import download_model
from .pipeline import WorkspaceSummary, annotate_dataset
from .review import apply_human_decision, load_human_decision, render_review_site
from .release_validator import validate_release
from .workspace import LegacyWorkspaceError, WorkspaceStore


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
model_app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.add_typer(model_app, name="model")


def serve_review_app(work_dir: Path, host: str, port: int) -> None:
    """Lazily import the optional HTTP stack and run one workspace-bound app."""
    import uvicorn
    from .review_server import create_review_app

    uvicorn.run(create_review_app(work_dir), host=host, port=port)


def parse_episode_indices(value: str) -> list[int]:
    """Parse a deliberately whitespace-free comma-separated index list."""
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError("episodes must be a nonempty comma-separated integer list without spaces")
    parts = value.split(",")
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("episodes must contain only non-negative decimal integers")
    result = [int(part) for part in parts]
    if len(set(result)) != len(result):
        raise ValueError("episode indices must be unique")
    return result


def _config(path: Path):
    try:
        return load_config(path)
    except Exception:
        typer.echo("Invalid configuration.", err=True)
        raise typer.Exit(2)


@app.command("inspect")
def inspect_command(
    config: Path,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    cfg = _config(config)
    try:
        dataset = inspect_dataset(cfg)
    except Exception:
        typer.echo("Dataset inspection failed.", err=True)
        raise typer.Exit(1)
    facts = {
        "dataset_version": dataset.version,
        "fps": dataset.fps,
        "cameras": dataset.camera_keys,
        "episodes": len(dataset.episodes),
        "frames": sum(item.length for item in dataset.episodes),
        "inspection": "OK",
    }
    if as_json:
        typer.echo(json.dumps(facts, sort_keys=True, separators=(",", ":")))
        return
    typer.echo(f"version: {dataset.version}")
    typer.echo(f"dataset_version: {dataset.version}")
    typer.echo(f"fps: {dataset.fps}")
    typer.echo(f"cameras: {', '.join(dataset.camera_keys)}")
    typer.echo(f"episodes: {len(dataset.episodes)}")
    typer.echo(f"frames: {sum(item.length for item in dataset.episodes)}")
    typer.echo("inspection: OK")


@app.command("annotate")
def annotate_command(
    config: Path,
    max_concurrency: int = typer.Option(1, min=1),
    episodes: str | None = typer.Option(None),
) -> None:
    cfg = _config(config)
    try:
        selected = None if episodes is None else parse_episode_indices(episodes)
    except ValueError:
        typer.echo("Invalid --episodes value.", err=True)
        raise typer.Exit(2)
    try:
        summary = asyncio.run(annotate_dataset(cfg, max_concurrency, selected))
    except Exception:
        typer.echo("Annotation failed.", err=True)
        raise typer.Exit(1)
    typer.echo(_summary_text(summary))
    if summary.failed:
        raise typer.Exit(1)


@app.command("status")
def status_command(work_dir: Path, as_json: bool = typer.Option(False, "--json")) -> None:
    try:
        raw = WorkspaceStore(work_dir).summary()
        summary = WorkspaceSummary.from_store_summary(raw)
    except LegacyWorkspaceError:
        typer.echo(
            "Legacy coarse-v4 workspace is read-only: preserve its JSON for manual audit "
            "and create a new workspace.",
            err=True,
        )
        raise typer.Exit(1)
    except Exception:
        typer.echo("Workspace status could not be read.", err=True)
        raise typer.Exit(1)
    if as_json:
        typer.echo(json.dumps(raw, sort_keys=True, separators=(",", ":")))
    else:
        typer.echo(_summary_text(summary))


@app.command("review")
def review_command(
    work_dir: Path,
    decision_path: Path | None = typer.Option(None, "--apply"),
    serve: bool = typer.Option(False, "--serve"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port", min=1, max=65535),
) -> None:
    if serve and decision_path is not None:
        typer.echo("--serve and --apply cannot be used together.", err=True)
        raise typer.Exit(2)
    if serve:
        try:
            serve_review_app(work_dir, host, port)
        except Exception:
            typer.echo("Visual review server failed.", err=True)
            raise typer.Exit(1)
        return
    if decision_path is None:
        try:
            page = render_review_site(work_dir)
        except Exception:
            typer.echo("Review site generation failed.", err=True)
            raise typer.Exit(1)
        typer.echo(str(page))
        return
    try:
        decision = load_human_decision(decision_path)
    except Exception:
        typer.echo("Invalid human decision file.", err=True)
        raise typer.Exit(2)
    try:
        accepted = apply_human_decision(work_dir, decision.episode_index, decision)
    except Exception:
        typer.echo("Human decision was rejected.", err=True)
        raise typer.Exit(1)
    typer.echo(f"accepted episode {accepted.episode_index}")


@app.command("convert")
def convert_command(
    work_dir: Path,
    output: Path = typer.Option(..., "--output"),
    accepted_only: bool = typer.Option(False, "--accepted-only"),
) -> None:
    try:
        report = convert_dataset(work_dir, output, accepted_only=accepted_only)
    except Exception:
        typer.echo("Dataset conversion failed.", err=True)
        raise typer.Exit(1)
    typer.echo(f"converted episodes={report.episode_count} frames={report.frame_count} output={report.output}")


@app.command("validate")
def validate_command(
    path: Path,
    source: Path | None = typer.Option(None, "--source"),
    allow_legacy_sampled_image_stats: bool = typer.Option(False, "--allow-legacy-sampled-image-stats"),
    deep_video_stats: bool = typer.Option(True, "--deep-video-stats/--no-deep-video-stats"),
) -> None:
    if allow_legacy_sampled_image_stats and deep_video_stats:
        typer.echo("Legacy sampled image stats require --no-deep-video-stats.", err=True)
        raise typer.Exit(2)
    try:
        report = validate_release(
            path,
            source=source,
            allow_legacy_sampled_image_stats=allow_legacy_sampled_image_stats,
            deep_video_stats=deep_video_stats,
        )
    except Exception:
        typer.echo("Release validation failed.", err=True)
        raise typer.Exit(1)
    skipped = ",".join(report.skipped_checks) if report.skipped_checks else "none"
    typer.echo(
        f"valid episodes={report.episode_count} frames={report.frame_count} path={report.path} "
        f"validation_level={report.validation_level} skipped_checks={skipped}"
    )


@app.command("evaluate")
def evaluate_command(
    work_dir: Path,
    golden: Path = typer.Option(..., "--golden"),
    output: Path = typer.Option(..., "--output"),
    obvious_error_threshold_seconds: float = typer.Option(
        1.0, "--obvious-error-threshold-seconds", min=0.000001
    ),
) -> None:
    try:
        metrics = evaluate_complete(
            work_dir,
            golden,
            obvious_error_threshold_seconds=obvious_error_threshold_seconds,
        )
        report = write_evaluation_report(output, metrics)
    except Exception:
        typer.echo("Evaluation failed.", err=True)
        raise typer.Exit(1)
    gates = report["launch_gates"]
    typer.echo(
        f"evaluated episodes={metrics.episode_count} boundaries={metrics.aligned_boundary_count} "
        f"launch_gates={'PASS' if gates['all_passed'] else 'FAIL'} output={output}"
    )
    if not gates["all_passed"]:
        raise typer.Exit(1)


@model_app.command("download")
def model_download(
    repo: str = typer.Option("Qwen/Qwen3.8-27B"),
    local_dir: Path = typer.Option(Path("/mnt/data/user/zhoukr/models/Qwen3.8-27B")),
    revision: str | None = typer.Option(None),
    max_workers: int = typer.Option(8, min=1),
) -> None:
    try:
        install = download_model(repo, local_dir, revision, max_workers=max_workers)
    except (TypeError, ValueError):
        typer.echo("Invalid model download arguments.", err=True)
        raise typer.Exit(2)
    except Exception:
        typer.echo("Model download failed.", err=True)
        raise typer.Exit(1)
    typer.echo(f"revision: {install.revision}")
    typer.echo(f"local_path: {install.local_path}")


def _summary_text(summary: WorkspaceSummary) -> str:
    return (
        f"total={summary.total} pending={summary.pending} coarse_done={summary.coarse_done} "
        f"refine_done={summary.refine_done} accepted={summary.accepted} "
        f"needs_review={summary.needs_review} failed={summary.failed}"
    )


if __name__ == "__main__":
    app()
