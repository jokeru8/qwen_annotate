from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from qwen_annotate.model_manager import (
    ModelInstall,
    RevisionResolutionError,
    download_model,
    resolve_revision,
    verify_model,
)


SHA = "a" * 40


class FakeResponse:
    def __init__(self, payload: Any, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> Any:
        return self.payload


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.urls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.urls.append(url)
        return self.response


def test_exact_sha_resolution_never_touches_client() -> None:
    client = FakeClient(FakeResponse({"sha": "b" * 40}))
    assert resolve_revision("Qwen/Qwen3.8-27B", SHA, client=client) == SHA
    assert client.urls == []


def test_mutable_revision_is_url_encoded_and_resolved() -> None:
    client = FakeClient(FakeResponse({"sha": "b" * 40}))
    assert resolve_revision("Qwen/Qwen3.8-27B", "refs/pr #1", client=client) == "b" * 40
    assert client.urls == [
        "https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/refs%2Fpr%20%231"
    ]


def test_owned_http_client_disables_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    class OwnedClient(FakeClient):
        def __enter__(self) -> "OwnedClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def factory(*, trust_env: bool) -> OwnedClient:
        seen.append(trust_env)
        return OwnedClient(FakeResponse({"sha": SHA}))

    monkeypatch.setattr("qwen_annotate.model_manager.httpx.Client", factory)
    assert resolve_revision("Qwen/Qwen3.8-27B") == SHA
    assert seen == [False]


@pytest.mark.parametrize(
    "payload",
    [[], {}, {"sha": "a" * 39}, {"sha": "A" * 40}, {"sha": 123}],
)
def test_malformed_revision_response_is_rejected(payload: Any) -> None:
    with pytest.raises(RevisionResolutionError, match="valid immutable SHA"):
        resolve_revision("Qwen/Qwen3.8-27B", client=FakeClient(FakeResponse(payload)))


def test_http_status_error_is_safe() -> None:
    request = httpx.Request("GET", "https://huggingface.co/redacted")
    response = httpx.Response(401, request=request, text="secret-token")
    client = FakeClient(FakeResponse({}, error=httpx.HTTPStatusError("secret-token", request=request, response=response)))
    with pytest.raises(RevisionResolutionError) as caught:
        resolve_revision("Qwen/Qwen3.8-27B", client=client)
    assert "secret-token" not in str(caught.value)
    assert "401" in str(caught.value)


@pytest.mark.parametrize("repo", ["", "/model", "owner/", "a/b/c", "../model", "owner/model?token=x", "owner\\model"])
def test_invalid_repo_is_rejected(repo: str) -> None:
    with pytest.raises(ValueError, match="repo"):
        resolve_revision(repo, SHA)


@pytest.mark.parametrize("revision", ["", "A" * 40, "abc", "g" * 40])
def test_verify_requires_exact_lowercase_sha(tmp_path: Path, revision: str) -> None:
    with pytest.raises(ValueError, match="40-character lowercase"):
        verify_model("Qwen/Qwen3.8-27B", revision, tmp_path, runner=lambda *_: None)


def test_download_commands_proxy_isolation_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    for key in proxy_keys:
        monkeypatch.setenv(key, "socks5://proxy:1080")
    monkeypatch.setenv("MODEL_MANAGER_SENTINEL", "kept")
    before = os.environ.copy()
    seen: list[tuple[list[str], dict[str, str]]] = []

    def runner(args: list[str], env: dict[str, str]) -> None:
        seen.append((args, env))
        if len(seen) == 1:
            env["MUTATED_BY_RUNNER"] = "yes"

    relative = tmp_path / ".." / tmp_path.name / "model"
    result = download_model(
        "Qwen/Qwen3.8-27B", relative, revision=SHA, runner=runner, max_workers=7
    )
    absolute = str(relative.resolve())
    assert seen[0][0] == [
        "hf", "download", "Qwen/Qwen3.8-27B", "--revision", SHA,
        "--local-dir", absolute, "--max-workers", "7",
    ]
    assert seen[1][0] == [
        "hf", "cache", "verify", "Qwen/Qwen3.8-27B", "--revision", SHA,
        "--local-dir", absolute, "--fail-on-missing-files",
    ]
    for _, env in seen:
        assert all(key not in env for key in proxy_keys)
        assert env["MODEL_MANAGER_SENTINEL"] == "kept"
    assert seen[0][1] is not seen[1][1]
    assert "MUTATED_BY_RUNNER" not in seen[1][1]
    assert os.environ == before

    assert result.repo == "Qwen/Qwen3.8-27B"
    assert result.revision == SHA
    assert result.local_path == relative.resolve()
    assert result.verified_at.tzinfo == UTC
    metadata = json.loads((relative.resolve() / "model-install.json").read_text())
    assert metadata == result.to_dict()
    assert metadata["verified_at"].endswith("Z")
    assert ModelInstall.from_dict(metadata) == result


@pytest.mark.parametrize("max_workers", [True, False, 0, -1, 1.5, "8"])
def test_max_workers_must_be_positive_non_bool(tmp_path: Path, max_workers: Any) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        download_model("Qwen/Qwen3.8-27B", tmp_path, revision=SHA, max_workers=max_workers)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_command_failure_leaves_no_metadata(tmp_path: Path, failure_call: int) -> None:
    calls = 0

    def runner(args: list[str], env: dict[str, str]) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise subprocess.CalledProcessError(1, args)

    with pytest.raises(subprocess.CalledProcessError):
        download_model("Qwen/Qwen3.8-27B", tmp_path / "model", revision=SHA, runner=runner)
    assert not (tmp_path / "model" / "model-install.json").exists()


def test_failed_rerun_preserves_previous_success_metadata(tmp_path: Path) -> None:
    target = tmp_path / "model"
    first = download_model("Qwen/Qwen3.8-27B", target, revision=SHA, runner=lambda *_: None)
    original = (target / "model-install.json").read_bytes()

    def fail(*_: object) -> None:
        raise subprocess.CalledProcessError(2, ["hf"])

    with pytest.raises(subprocess.CalledProcessError):
        download_model("Qwen/Qwen3.8-27B", target, revision=SHA, runner=fail)
    assert (target / "model-install.json").read_bytes() == original
    assert ModelInstall.from_dict(json.loads(original)) == first


def test_atomic_metadata_cleanup_when_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "model"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("qwen_annotate.model_manager.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        download_model("Qwen/Qwen3.8-27B", target, revision=SHA, runner=lambda *_: None)
    assert not (target / "model-install.json").exists()
    assert list(target.glob(".model-install.json.*.tmp")) == []


def test_default_runner_uses_check_env_and_no_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(args: list[str], **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("qwen_annotate.model_manager.subprocess.run", fake_run)
    verify_model("Qwen/Qwen3.8-27B", SHA, tmp_path)
    assert calls[0][1]["check"] is True
    assert isinstance(calls[0][1]["env"], dict)
    assert "shell" not in calls[0][1]


def test_install_validation_rejects_naive_time_and_relative_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ModelInstall("Qwen/Qwen3.8-27B", SHA, Path("relative"), datetime.now())


def test_no_shell_metacharacters_are_interpreted(tmp_path: Path) -> None:
    seen: list[list[str]] = []
    target = tmp_path / "model;touch should-not-exist"
    download_model("Qwen/Qwen3.8-27B", target, revision=SHA, runner=lambda args, env: seen.append(args))
    assert str(target.resolve()) in seen[0]
    assert not (tmp_path / "should-not-exist").exists()
    assert re.fullmatch(r"[0-9a-f]{40}", SHA)
