# Visual Manual Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local multi-camera review UI that safely accepts reviewed model candidates and explicitly confirmed manual takeover annotations into the authoritative Qwen workspace.

**Architecture:** Add one strict human-decision service over `WorkspaceStore`, then expose bounded read/video/write endpoints through a FastAPI app. Serve a dependency-free browser UI whose synchronized videos and timeline consume only that API; keep static review and CLI decision import backward compatible.

**Tech Stack:** Python 3.12, Pydantic 2, FastAPI, Uvicorn, native HTML/CSS/JavaScript, Pytest

**Spec:** `docs/superpowers/specs/2026-08-23-visual-manual-review-design.md`

## Global Constraints

- The server is bound to exactly one explicit workspace and defaults to `127.0.0.1`.
- `pending`, `failed`, and accepted corrections require explicit takeover confirmation.
- Every write revalidates fingerprints, optimistic state, mode, and annotation constraints.
- Source datasets are read-only and browser inputs never resolve filesystem paths.
- Existing static review and portable decision import behavior remains compatible.
- New production behavior follows red-green-refactor TDD.

---

### Task 1: Human takeover state contract

**Files:**
- Modify: `src/robo_annotate/review.py`
- Modify: `src/robo_annotate/workspace.py`
- Modify: `tests/test_review.py`
- Modify: `tests/test_workspace.py`

**Interfaces:**
- Produces: `ManualDecision`, `apply_manual_decision(work_dir, decision, *, services=None) -> EpisodeRecord`
- Produces: guarded human transitions `pending|failed|needs_review|accepted -> accepted`

- [x] Write failing tests proving pending/failed require `takeover_confirmed=True`, stale status/time is rejected, failed diagnostics are copied into audit, and accepted correction is audited.
- [x] Run `uv run pytest tests/test_review.py tests/test_workspace.py -q` and verify failures are caused by the absent contract.
- [x] Add strict decision fields: expected status, expected updated timestamp, confirmation, annotation, and optional bounded note.
- [x] Centralize fingerprint/constraint validation and append a complete human audit without deleting attempts.
- [x] Permit only human-audited direct/same-status transitions and extend outbox validation for human event names.
- [x] Run the focused tests and verify they pass.
- [x] Commit with `git commit -m "feat: add audited manual takeover decisions"`.

### Task 2: Workspace transaction portability

**Files:**
- Modify: `src/robo_annotate/workspace.py`
- Modify: `tests/test_workspace.py`

**Interfaces:**
- Consumes: `WorkspaceStore.save_episode_transactional(record)`
- Produces: rollback on filesystems that reject hard links

- [x] Write a failing test that makes `os.link` raise the FUSE-style unsupported-operation error and then forces summary refresh failure.
- [x] Run that test and verify the authoritative episode is not currently restored.
- [x] Replace hard-link-only rollback with a bounded regular-file backup copied under the locked workspace logs directory, preserving no-follow and fsync behavior.
- [x] Run transactional and workspace tests and verify rollback and cleanup.
- [x] Commit with `git commit -m "fix: make human decision rollback FUSE-safe"`.

### Task 3: Review read model and video responses

**Files:**
- Create: `src/robo_annotate/review_server.py`
- Create: `tests/test_review_server.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `ReviewRuntime.open(work_dir)`, `session_payload()`, `episode_payload(index)`, `video_path(index, camera)`
- Produces: `create_review_app(work_dir) -> FastAPI`

- [x] Write failing API tests for session metadata, all-status episode listings, safe detail payloads, invalid episode/camera, traversal input, and byte Range responses.
- [x] Run `uv run pytest tests/test_review_server.py -q` and verify import/route failures.
- [x] Add FastAPI/Uvicorn dependencies with `uv add fastapi uvicorn` so the lock file is reproducible.
- [x] Implement a runtime that loads manifest/config/dataset once, validates identity, and returns explicitly constructed safe payloads.
- [x] Implement regular-file video streaming for full, open-ended, suffix, and bounded byte ranges; return 416 for malformed/unsatisfiable ranges.
- [x] Run API tests and commit with `git commit -m "feat: expose bounded review and video API"`.

### Task 4: Decision endpoint and conflict mapping

**Files:**
- Modify: `src/robo_annotate/review_server.py`
- Modify: `tests/test_review_server.py`

**Interfaces:**
- Consumes: `ManualDecision`, `apply_manual_decision`
- Produces: `POST /api/episodes/{index}/decision`

- [x] Write failing tests for takeover-required 409, stale-record 409, invalid annotation 422, fingerprint mismatch 409, successful review acceptance, and successful failed takeover.
- [x] Run the focused tests and verify each missing route/branch fails.
- [x] Implement strict request parsing and map known validation conflicts to stable safe API error codes.
- [x] Re-read the saved record after success and return its safe episode payload.
- [x] Run tests and commit with `git commit -m "feat: add safe visual review writeback API"`.

### Task 5: Synchronized multi-camera browser UI

**Files:**
- Create: `src/robo_annotate/review_web/index.html`
- Create: `src/robo_annotate/review_web/app.js`
- Create: `src/robo_annotate/review_web/style.css`
- Create: `tests/test_review_web.py`
- Modify: `src/robo_annotate/review_server.py`

**Interfaces:**
- Consumes: `/api/session`, `/api/episodes`, episode detail/video URLs, decision endpoint
- Produces: responsive local annotation UI and pure exported JS helpers for frame/boundary calculations

- [x] Write failing tests that execute pure JS helpers for frame/time conversion, boundary insertion/removal, current segment, and decision payload gating; add HTTP tests for packaged assets and CSP.
- [x] Run focused tests and verify missing assets/helpers fail.
- [x] Build the three-pane layout with status filters, episode list, primary camera, promotable synchronized cameras, task panel, evidence, and keyboard help.
- [x] Implement one master playback clock, drift correction, exact paused frame stepping, shared seek, and frame-number display.
- [x] Implement colored segment timeline, editable markers, start-subtask selection, validation feedback, browser-memory drafts, and immutable model/source context.
- [x] Implement explicit confirmation dialogs for pending/failed takeover and accepted correction, then submit the optimistic decision payload.
- [x] Run UI/API tests and commit with `git commit -m "feat: add synchronized visual annotation interface"`.

### Task 6: CLI and operator documentation

**Files:**
- Modify: `src/robo_annotate/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `docs/operations.md`

**Interfaces:**
- Produces: `Robo-annotate review WORK_DIR --serve --host 127.0.0.1 --port 8765`

- [x] Write failing CLI tests proving `--serve` delegates to Uvicorn, rejects incompatible `--apply`, and keeps default static generation.
- [x] Run `uv run pytest tests/test_cli.py -q` and verify failures.
- [x] Implement lazy server imports and safe host/port options without changing import-time GPU/network behavior.
- [x] Document launch, SSH port forwarding, status semantics, takeover confirmation, keyboard controls, and the separate convert/validate publication flow.
- [x] Run CLI tests and commit with `git commit -m "docs: add visual review workflow"`.

### Task 7: End-to-end verification

**Files:**
- Modify only files required by defects reproduced in this task, with a failing regression test first.

**Interfaces:**
- Consumes: the complete local review workflow
- Produces: verified release-ready implementation

- [x] Run `uv run pytest tests/test_review.py tests/test_workspace.py tests/test_review_server.py tests/test_review_web.py tests/test_cli.py -q`.
- [x] Run `uv run pytest -q` and resolve each defect through a new failing regression test before changing production code.
- [x] Run `uv run Robo-annotate --help` and `uv run Robo-annotate review --help` as CLI smoke tests.
- [x] Start the server against a disposable fixture, request `/`, `/api/session`, one episode, and a byte range, then stop it cleanly.
- [x] Inspect `git diff --check`, `git status --short`, and the final diff for secrets, source writes, unrelated changes, or generated artifacts.
- [x] Commit verification fixes, if any, with a narrowly scoped message.
