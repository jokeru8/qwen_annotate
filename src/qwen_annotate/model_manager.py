"""Install Hugging Face models at verified, immutable revisions."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_REPO_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_METADATA_NAME = "model-install.json"

Runner = Callable[[list[str], dict[str, str]], object]


class RevisionResolutionError(RuntimeError):
    """A mutable Hugging Face revision could not be resolved safely."""


@dataclass(frozen=True, slots=True)
class ModelInstall:
    """Auditable description of a successfully verified local model."""

    repo: str
    revision: str
    local_path: Path
    verified_at: datetime

    def __post_init__(self) -> None:
        _validate_repo(self.repo)
        _validate_sha(self.revision)
        if not isinstance(self.local_path, Path) or not self.local_path.is_absolute():
            raise ValueError("local_path must be an absolute Path")
        if not isinstance(self.verified_at, datetime):
            raise ValueError("verified_at must be a datetime")
        offset = self.verified_at.utcoffset()
        if self.verified_at.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("verified_at must have an unambiguous UTC timezone")

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON-compatible metadata representation."""
        timestamp = self.verified_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return {
            "repo": self.repo,
            "revision": self.revision,
            "local_path": str(self.local_path),
            "verified_at": timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelInstall:
        """Load metadata written by :func:`download_model`."""
        if not isinstance(payload, Mapping):
            raise ValueError("model install metadata must be an object")
        expected = {"repo", "revision", "local_path", "verified_at"}
        if set(payload) != expected or not all(isinstance(payload[key], str) for key in expected):
            raise ValueError("model install metadata has invalid fields")
        try:
            verified_at = datetime.fromisoformat(payload["verified_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("model install metadata has an invalid verified_at") from exc
        return cls(
            repo=payload["repo"],
            revision=payload["revision"],
            local_path=Path(payload["local_path"]),
            verified_at=verified_at,
        )


def resolve_revision(
    repo: str,
    revision: str = "main",
    *,
    client: Any | None = None,
) -> str:
    """Resolve a model revision to an exact lowercase 40-character commit SHA.

    Exact SHAs are accepted without a network call. When this function owns the
    HTTP client it disables environment-derived proxies with ``trust_env=False``.
    """
    _validate_repo(repo)
    _validate_revision_name(revision)
    if _SHA_PATTERN.fullmatch(revision):
        return revision

    repo_path = "/".join(quote(component, safe="") for component in repo.split("/"))
    url = (
        f"https://huggingface.co/api/models/{repo_path}/revision/"
        f"{quote(revision, safe='')}"
    )
    if client is None:
        try:
            with httpx.Client(trust_env=False) as owned_client:
                return _request_revision(owned_client, url)
        except RevisionResolutionError:
            raise
        except httpx.HTTPError as exc:
            raise _safe_http_error(exc) from None
    return _request_revision(client, url)


def download_model(
    repo: str,
    local_path: Path,
    revision: str | None = None,
    *,
    runner: Runner | None = None,
    client: Any | None = None,
    max_workers: int = 8,
) -> ModelInstall:
    """Download, verify, and atomically record an immutable model install.

    ``runner`` is called as ``runner(argv, child_env)``. Each call receives a
    fresh proxy-isolated environment mapping and no command is run via a shell.
    """
    _validate_repo(repo)
    if type(max_workers) is not int or max_workers <= 0:
        raise ValueError("max_workers must be a positive non-boolean integer")
    target = _absolute_local_path(local_path)
    sha = resolve_revision(repo, revision or "main", client=client)
    command_runner = runner or _default_runner

    target.mkdir(parents=True, exist_ok=True)
    command_runner(
        [
            "hf",
            "download",
            repo,
            "--revision",
            sha,
            "--local-dir",
            str(target),
            "--max-workers",
            str(max_workers),
        ],
        _proxy_isolated_env(),
    )
    verify_model(repo, sha, target, runner=command_runner)

    install = ModelInstall(
        repo=repo,
        revision=sha,
        local_path=target,
        verified_at=datetime.now(UTC),
    )
    _write_install_metadata(install)
    return install


def verify_model(
    repo: str,
    revision: str,
    local_path: Path,
    *,
    runner: Runner | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Verify every repository file against an immutable revision."""
    _validate_repo(repo)
    _validate_sha(revision)
    target = _absolute_local_path(local_path)
    command_runner = runner or _default_runner
    command_runner(
        [
            "hf",
            "cache",
            "verify",
            repo,
            "--revision",
            revision,
            "--local-dir",
            str(target),
            "--fail-on-missing-files",
        ],
        _proxy_isolated_env(env),
    )


def _request_revision(client: Any, url: str) -> str:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise _safe_http_error(exc) from None
    except Exception as exc:
        raise RevisionResolutionError(
            f"revision request failed ({type(exc).__name__})"
        ) from None
    try:
        payload = response.json()
    except Exception:
        raise RevisionResolutionError(
            "revision response did not contain a valid JSON object with a valid immutable SHA"
        ) from None
    if not isinstance(payload, dict) or not isinstance(payload.get("sha"), str):
        raise RevisionResolutionError(
            "revision response did not contain a valid JSON object with a valid immutable SHA"
        )
    sha = payload["sha"]
    if not _SHA_PATTERN.fullmatch(sha):
        raise RevisionResolutionError("revision response did not contain a valid immutable SHA")
    return sha


def _safe_http_error(exc: httpx.HTTPError) -> RevisionResolutionError:
    if isinstance(exc, httpx.HTTPStatusError):
        return RevisionResolutionError(
            f"revision request returned HTTP status {exc.response.status_code}"
        )
    return RevisionResolutionError(f"revision request failed ({type(exc).__name__})")


def _default_runner(args: list[str], env: dict[str, str]) -> object:
    return subprocess.run(args, check=True, env=env)


def _proxy_isolated_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    child = dict(os.environ if source is None else source)
    for key in _PROXY_KEYS:
        child.pop(key, None)
    return child


def _write_install_metadata(install: ModelInstall) -> None:
    destination = install.local_path / _METADATA_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{_METADATA_NAME}.", suffix=".tmp", dir=install.local_path
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(install.to_dict(), stream, ensure_ascii=True, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_repo(repo: str) -> None:
    if not isinstance(repo, str):
        raise ValueError("repo must be a valid Hugging Face repository identifier")
    components = repo.split("/")
    if not 1 <= len(components) <= 2 or any(
        not _REPO_COMPONENT.fullmatch(component)
        or component in {".", ".."}
        or ".." in component
        or "--" in component
        for component in components
    ):
        raise ValueError("repo must be a valid Hugging Face repository identifier")


def _validate_revision_name(revision: str) -> None:
    if not isinstance(revision, str) or not revision or any(ord(char) < 32 for char in revision):
        raise ValueError("revision must be a non-empty printable string")


def _validate_sha(revision: str) -> None:
    if not isinstance(revision, str) or not _SHA_PATTERN.fullmatch(revision):
        raise ValueError("revision must be an exact 40-character lowercase hexadecimal SHA")


def _absolute_local_path(local_path: Path) -> Path:
    if not isinstance(local_path, (str, os.PathLike)):
        raise ValueError("local_path must be a filesystem path")
    raw = os.fspath(local_path)
    if not raw or "\x00" in raw:
        raise ValueError("local_path must be a non-empty valid filesystem path")
    return Path(raw).expanduser().resolve()
