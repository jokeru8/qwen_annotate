import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from qwen_annotate.review_server import create_review_app
from tests.test_review import _workspace


def _run_js(expression: str) -> object:
    module = Path("src/qwen_annotate/review_web/app.js").resolve().as_uri()
    script = f"import({json.dumps(module)}).then(m => console.log(JSON.stringify({expression})))"
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_timeline_helpers_use_boundary_as_first_frame_of_next_subtask() -> None:
    result = _run_js("({"
        "frames:[m.frameFromTime(1.26,20,100),m.frameFromTime(99,20,100)],"
        "segments:[m.segmentAt(9,0,[10,20],3),m.segmentAt(10,0,[10,20],3),"
        "m.segmentAt(20,0,[10,20],3)]})")
    assert result == {"frames": [25, 99], "segments": [0, 1, 2]}


def test_boundary_edit_helpers_keep_valid_strict_order() -> None:
    result = _run_js("({"
        "insert:m.insertBoundary([30,10],20,100),"
        "duplicate:m.insertBoundary([10,20],20,100),"
        "remove:m.removeNearestBoundary([10,20,30],22),"
        "invalid:m.insertBoundary([10],0,100)})")
    assert result == {
        "insert": [10, 20, 30],
        "duplicate": [10, 20],
        "remove": [10, 30],
        "invalid": [10],
    }


def test_decision_builder_enforces_explicit_takeover() -> None:
    result = _run_js("(() => {"
        "const base={episode_index:2,status:'pending',updated_at:'2026-01-01T00:00:00Z',"
        "source_fingerprint:'a',run_fingerprint:'b',mode:'complete'};"
        "let blocked=false;try{m.buildDecision(base,0,[10],false,'')}catch(e){blocked=true};"
        "return {blocked,ok:m.buildDecision(base,0,[10],true,'manual')};})()")
    assert result["blocked"] is True
    assert result["ok"]["takeover_confirmed"] is True
    assert result["ok"]["expected_status"] == "pending"
    assert result["ok"]["boundaries"] == [10]


def test_draft_validation_reports_mode_count_order_and_short_segments() -> None:
    result = _run_js("({"
        "complete:m.validateDraft({mode:'complete',subtaskCount:3,length:100,minSegmentFrames:10},1,[5]),"
        "dagger:m.validateDraft({mode:'dagger_patch',subtaskCount:4,length:100,minSegmentFrames:10},1,[30]),"
        "valid:m.validateDraft({mode:'complete',subtaskCount:3,length:100,minSegmentFrames:10},0,[20,70])})")
    assert result["complete"] == ["complete_start_index", "complete_boundary_count", "segment_too_short"]
    assert result["dagger"] == ["dagger_suffix_length"]
    assert result["valid"] == []


def test_pointer_and_camera_helpers_drive_drag_and_promotion_behavior() -> None:
    result = _run_js("({"
        "frames:[m.frameFromPointer(100,300,200,101),m.frameFromPointer(499,300,200,101)],"
        "actions:[m.cameraClickAction('wrist','eye'),m.cameraClickAction('eye','eye')]})")
    assert result == {"frames": [0, 100], "actions": ["promote", "toggle_play"]}


def test_episode_drafts_are_copied_and_restored_independently() -> None:
    result = _run_js("(() => {const drafts=new Map();const boundaries=[10];"
        "m.rememberDraft(drafts,2,{start:1,boundaries,note:'check',takeover:true,status:'pending',updated_at:'t1'});"
        "boundaries.push(20);return m.restoreDraft(drafts,{episode_index:2,status:'pending',updated_at:'t1',candidate_annotation:null});})()")
    assert result == {"start": 1, "boundaries": [10], "note": "check", "takeover": True}


def test_takeover_draft_is_reset_when_authoritative_record_version_changes() -> None:
    result = _run_js("(() => {const drafts=new Map();"
        "m.rememberDraft(drafts,2,{start:1,boundaries:[10],note:'check',takeover:true,status:'pending',updated_at:'t1'});"
        "return m.restoreDraft(drafts,{episode_index:2,status:'failed',updated_at:'t2',candidate_annotation:{start_subtask_index:0,boundaries:[]}});})()")
    assert result == {"start": 1, "boundaries": [10], "note": "check", "takeover": False}


def test_save_response_only_applies_to_same_selection_generation() -> None:
    result = _run_js("[m.shouldApplySaveResponse(2,2,7,7),m.shouldApplySaveResponse(3,2,8,7)]")
    assert result == [True, False]


def test_committed_save_is_applied_before_optional_refresh() -> None:
    result = _run_js("(() => {const view={detail:{episode_index:2},selectionGeneration:7,takeover:true,drafts:new Map([[2,{}]])};"
        "const applied=m.applyCommittedResponse(view,{episode_index:2,status:'accepted'},2,7);"
        "return {applied,status:view.detail.status,takeover:view.takeover,hasDraft:view.drafts.has(2)};})()")
    assert result == {"applied": True, "status": "accepted", "takeover": False, "hasDraft": False}


def test_selection_completion_unlocks_only_current_generation() -> None:
    result = _run_js("(() => {const view={loading:true,selectionGeneration:4};"
        "return [m.finishSelection(view,3),view.loading,m.finishSelection(view,4),view.loading];})()")
    assert result == [False, True, True, False]


def test_web_assets_are_served_with_csp_and_no_directory_listing(tmp_path: Path) -> None:
    work, _, _, _, services, _ = _workspace(tmp_path)
    client = TestClient(create_review_app(work, services=services))

    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-security-policy"].startswith("default-src 'self'")
    assert "Qwen 自动标注复核" in page.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/style.css").status_code == 200
    assert client.get("/assets/../review_server.py").status_code == 404
