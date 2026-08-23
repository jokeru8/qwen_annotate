export function frameFromTime(time, fps, length) {
  if (!Number.isFinite(time) || !Number.isFinite(fps) || length <= 0) return 0;
  return Math.max(0, Math.min(length - 1, Math.round(time * fps)));
}

export function segmentAt(frame, startSubtask, boundaries, subtaskCount) {
  let index = startSubtask;
  for (const boundary of boundaries) {
    if (frame < boundary) break;
    index += 1;
  }
  return Math.max(0, Math.min(subtaskCount - 1, index));
}

export function insertBoundary(boundaries, frame, length) {
  if (!Number.isInteger(frame) || frame <= 0 || frame >= length) return [...new Set(boundaries)].sort((a, b) => a - b);
  return [...new Set([...boundaries, frame])].sort((a, b) => a - b);
}

export function removeNearestBoundary(boundaries, frame) {
  if (!boundaries.length) return [];
  const nearest = boundaries.reduce((best, value) =>
    Math.abs(value - frame) < Math.abs(best - frame) ? value : best);
  return boundaries.filter(value => value !== nearest);
}

export function buildDecision(episode, startSubtask, boundaries, takeoverConfirmed, note) {
  if (["pending", "failed", "accepted"].includes(episode.status) && !takeoverConfirmed) {
    throw new Error("takeover_required");
  }
  return {
    episode_index: episode.episode_index,
    source_fingerprint: episode.source_fingerprint,
    run_fingerprint: episode.run_fingerprint,
    mode: episode.mode,
    expected_status: episode.status,
    expected_updated_at: episode.updated_at,
    takeover_confirmed: takeoverConfirmed,
    start_subtask_index: startSubtask,
    boundaries: [...boundaries],
    note,
  };
}

export function frameFromPointer(clientX, left, width, length) {
  if (width <= 0 || length <= 0) return 0;
  const ratio = Math.max(0, Math.min(1, (clientX - left) / width));
  return Math.round(ratio * (length - 1));
}

export function cameraClickAction(camera, primary) {
  return "toggle_play";
}

export function needsFrameCorrection(actualTime, targetTime, fps) {
  return Math.abs(actualTime - targetTime) * fps >= 0.5 - 1e-9;
}

function seekOneVideo(video, targetTime) {
  return new Promise((resolve, reject) => {
    const begin = () => {
      if (Math.abs(video.currentTime - targetTime) < 1e-9 && !video.seeking) {
        resolve();
        return;
      }
      const cleanup = () => {
        video.removeEventListener("seeked", complete);
        video.removeEventListener("error", fail);
      };
      const complete = () => { cleanup(); resolve(); };
      const fail = () => { cleanup(); reject(new Error("video_seek_failed")); };
      video.addEventListener("seeked", complete, {once: true});
      video.addEventListener("error", fail, {once: true});
      try { video.currentTime = targetTime; }
      catch (error) { cleanup(); reject(error); }
    };
    if (video.readyState === 0) video.addEventListener("loadedmetadata", begin, {once: true});
    else begin();
  });
}

export function seekVideosToFrame(videos, frame, fps) {
  const targetTime = frame / fps;
  return Promise.all([...videos].map(video => seekOneVideo(video, targetTime)));
}

export function validateDraft(context, startSubtask, boundaries) {
  const issues = [];
  if (!Number.isInteger(startSubtask) || startSubtask < 0 || startSubtask >= context.subtaskCount) issues.push("start_subtask_range");
  if (context.mode === "complete") {
    if (startSubtask !== 0) issues.push("complete_start_index");
    if (boundaries.length !== context.subtaskCount - 1) issues.push("complete_boundary_count");
  } else if (boundaries.length && boundaries.length !== context.subtaskCount - startSubtask - 1) {
    issues.push("dagger_suffix_length");
  }
  if (boundaries.some(value => !Number.isInteger(value) || value <= 0 || value >= context.length)) issues.push("boundary_range");
  if (boundaries.some((value, index) => index > 0 && value <= boundaries[index - 1])) issues.push("boundary_order");
  const points = [0, ...boundaries, context.length];
  if (points.some((value, index) => index > 0 && value - points[index - 1] < context.minSegmentFrames)) issues.push("segment_too_short");
  return [...new Set(issues)];
}

export function rememberDraft(drafts, episodeIndex, draft) {
  drafts.set(episodeIndex, {start: draft.start, boundaries: [...draft.boundaries], note: draft.note,
    takeover: draft.takeover, status: draft.status, updated_at: draft.updated_at});
}

export function restoreDraft(drafts, detail) {
  const saved = drafts.get(detail.episode_index);
  if (saved) return {start: saved.start, boundaries: [...saved.boundaries], note: saved.note,
    takeover: Boolean(saved.takeover && saved.status === detail.status && saved.updated_at === detail.updated_at)};
  return {start: detail.candidate_annotation?.start_subtask_index ?? 0,
    boundaries: [...(detail.candidate_annotation?.boundaries ?? [])], note: "", takeover: false};
}

export function shouldApplySaveResponse(currentIndex, savedIndex, currentGeneration, saveGeneration) {
  return currentIndex === savedIndex && currentGeneration === saveGeneration;
}

export function applyCommittedResponse(view, saved, savedIndex, saveGeneration) {
  if (!shouldApplySaveResponse(view.detail?.episode_index, savedIndex, view.selectionGeneration, saveGeneration)) return false;
  view.detail = saved; view.drafts.delete(savedIndex); view.takeover = false;
  return true;
}

export function finishSelection(view, generation) {
  if (generation !== view.selectionGeneration) return false;
  view.loading = false;
  return true;
}

const palette = ["#45c4a0", "#56a8e8", "#a98aef", "#ef9f62", "#df6f8f", "#8dbb54"];
const state = {session: null, episodes: [], detail: null, frame: 0, boundaries: [], start: 0,
  filter: "all", takeover: false, primary: null, videos: new Map(), syncing: false, drafts: new Map(),
  loading: false, saving: false, selectionGeneration: 0, seekGeneration: 0, syncLoopId: null};

const $ = id => document.getElementById(id);
const api = async (url, options) => {
  const response = await fetch(url, options);
  const payload = response.headers.get("content-type")?.includes("json") ? await response.json() : null;
  if (!response.ok) throw Object.assign(new Error(payload?.detail?.message || "请求失败"), {response, payload});
  return payload;
};

function setText(id, value) { $(id).textContent = value ?? "—"; }
function statusLabel(status) {
  return ({pending:"待标注",coarse_done:"粗标完成",refine_done:"精标完成",accepted:"已接受",
    needs_review:"需复核",failed:"失败"})[status] || status;
}

async function boot() {
  state.session = await api("/api/session");
  state.episodes = await api("/api/episodes");
  setText("workspace-summary", `${state.session.total_episodes} episodes · ${state.session.fps} fps · ${state.session.camera_keys.length} cameras`);
  setText("instruction", state.session.high_level_instruction);
  renderFilters(); renderEpisodes(); renderStartOptions(); bindControls();
  if (state.episodes.length) await selectEpisode(state.episodes[0].episode_index);
}

function renderFilters() {
  const root = $("status-filters"); root.replaceChildren();
  const values = ["all", "pending", "needs_review", "failed", "accepted", "coarse_done", "refine_done"];
  for (const value of values) {
    const button = document.createElement("button");
    const count = value === "all" ? state.episodes.length : (state.session.status_counts[value] || 0);
    button.textContent = `${value === "all" ? "全部" : statusLabel(value)} ${count}`;
    button.className = value === state.filter ? "active" : "";
    button.onclick = () => { state.filter = value; renderFilters(); renderEpisodes(); };
    root.append(button);
  }
}

function renderEpisodes() {
  const visible = state.episodes.filter(item => state.filter === "all" || item.status === state.filter);
  setText("episode-count", `${visible.length}/${state.episodes.length}`);
  const root = $("episode-list"); root.replaceChildren();
  for (const item of visible) {
    const button = document.createElement("button");
    button.className = `episode-row ${state.detail?.episode_index === item.episode_index ? "selected" : ""}`;
    const title = document.createElement("strong"); title.textContent = `Episode ${String(item.episode_index).padStart(6,"0")}`;
    const badge = document.createElement("span"); badge.className = `dot ${item.status}`; badge.textContent = statusLabel(item.status);
    button.disabled = state.loading || state.saving;
    button.append(title, badge); button.onclick = () => selectEpisode(item.episode_index); root.append(button);
  }
}

function renderStartOptions() {
  const select = $("start-subtask"); select.replaceChildren();
  state.session.subtasks.forEach((task, index) => {
    const option = document.createElement("option"); option.value = String(index);
    option.textContent = `${index}. ${task.text}`; select.append(option);
  });
}

async function selectEpisode(index) {
  if (state.saving) return;
  const generation=++state.selectionGeneration; state.loading=true; rememberCurrentDraft(); pauseAll();
  renderEpisodes(); if(state.detail)renderDecision();
  let detail;
  try { detail=await api(`/api/episodes/${index}`); }
  catch(error) { if(generation===state.selectionGeneration)setText("decision-message",`Episode 加载失败：${error.message}`); return; }
  finally { if(finishSelection(state,generation)){renderEpisodes();if(state.detail)renderDecision();} }
  if(generation!==state.selectionGeneration)return;
  state.detail = detail; state.frame = 0;
  const draft = restoreDraft(state.drafts, detail);
  state.start = draft.start; state.boundaries = draft.boundaries; state.takeover = draft.takeover;
  state.primary = state.session.primary_camera in state.detail.video_urls ? state.session.primary_camera : Object.keys(state.detail.video_urls)[0];
  $("start-subtask").value = String(state.start); $("decision-note").value = draft.note;
  $("frame-slider").max = String(state.detail.episode_length - 1);
  setText("episode-label", `EPISODE ${String(index).padStart(6,"0")}`); setText("task-title", state.detail.task);
  setText("status-badge", statusLabel(state.detail.status)); $("status-badge").className = `status-badge ${state.detail.status}`;
  renderEpisodes(); renderVideos(); renderEvidence(); renderDecision(); renderTimeline(); seek(0);
}

function renderVideos() {
  state.videos.clear(); const root = $("video-grid"); root.replaceChildren(); root.className = "video-grid";
  const cameras = Object.keys(state.detail.video_urls);
  for (const camera of cameras) {
    const card = document.createElement("button"); card.type = "button"; card.dataset.camera = camera; card.className = "video-card";
    const video = document.createElement("video"); video.src = state.detail.video_urls[camera]; video.preload = "metadata"; video.muted = true; video.playsInline = true;
    const label = document.createElement("span"); label.className = "camera-label"; label.textContent = camera;
    const frame = document.createElement("span"); frame.className = "camera-frame"; frame.textContent = "帧 0";
    card.append(video, label, frame); card.onclick = togglePlay;
    video.addEventListener("click", event => { event.preventDefault(); event.stopPropagation(); togglePlay(); });
    state.videos.set(camera, video); root.append(card);
  }
}

function primaryVideo() { return state.videos.get(state.primary); }
function syncFromMaster(master) {
  state.frame = frameFromTime(master.currentTime, state.session.fps, state.detail.episode_length);
  const targetTime = state.frame / state.session.fps;
  for (const video of state.videos.values()) {
    if (video !== master && !video.seeking && needsFrameCorrection(video.currentTime, targetTime, state.session.fps)) {
      video.currentTime = targetTime;
    }
  }
  updateFrameUI();
}
function stopSyncLoop() {
  if (state.syncLoopId !== null) cancelAnimationFrame(state.syncLoopId);
  state.syncLoopId = null;
}
function startSyncLoop() {
  stopSyncLoop();
  const tick = () => {
    const master = primaryVideo();
    if (!master || master.paused) { state.syncLoopId = null; return; }
    if (!state.syncing) syncFromMaster(master);
    state.syncLoopId = requestAnimationFrame(tick);
  };
  state.syncLoopId = requestAnimationFrame(tick);
}
async function seek(frame) {
  if (!state.detail) return; state.frame = Math.max(0, Math.min(state.detail.episode_length - 1, frame));
  const generation = ++state.seekGeneration; state.syncing = true; updateFrameUI();
  try { await seekVideosToFrame(state.videos.values(), state.frame, state.session.fps); }
  finally { if (generation === state.seekGeneration) { state.syncing = false; updateFrameUI(); } }
}
function pauseAll() { stopSyncLoop(); for (const video of state.videos.values()) video.pause(); }
async function togglePlay() {
  const master = primaryVideo(); if (!master) return;
  if (master.paused) { await seek(state.frame); await Promise.all([...state.videos.values()].map(video => video.play().catch(() => {}))); $("play-button").textContent = "❚❚"; startSyncLoop(); }
  else { pauseAll(); $("play-button").textContent = "▶"; }
}
function step(delta) { pauseAll(); $("play-button").textContent = "▶"; void seek(state.frame + delta); }
function manualSeek(frame) { pauseAll(); $("play-button").textContent = "▶"; void seek(frame); }

function updateFrameUI() {
  $("frame-slider").value = String(state.frame);
  setText("frame-output", `帧 ${state.frame} · ${(state.frame/state.session.fps).toFixed(3)}s`);
  const active = segmentAt(state.frame, state.start, state.boundaries, state.session.subtasks.length);
  [...$("subtask-list").children].forEach((item,index) => item.classList.toggle("current", index === active));
  for (const badge of $("video-grid").querySelectorAll(".camera-frame")) badge.textContent = `帧 ${state.frame}`;
  renderBoundaryList();
}

function renderTimeline() {
  const root = $("timeline"); root.replaceChildren();
  const points = [0, ...state.boundaries, state.detail.episode_length];
  for (let i=0; i<points.length-1; i++) {
    const segment = document.createElement("div"); const taskIndex = state.start + i;
    segment.className = "timeline-segment"; segment.style.width = `${(points[i+1]-points[i])/state.detail.episode_length*100}%`;
    segment.style.background = palette[taskIndex % palette.length]; segment.textContent = String(taskIndex);
    root.append(segment);
  }
  state.boundaries.forEach((frame,index) => {
    const handle=document.createElement("button"); handle.type="button"; handle.className="boundary-handle";
    handle.style.left=`${frame/state.detail.episode_length*100}%`;
    handle.setAttribute("aria-label",`拖动边界 ${index+1}，当前帧 ${frame}`);
    handle.disabled=!canEdit(); handle.onpointerdown=event=>beginBoundaryDrag(event,index,handle);
    handle.onkeydown=event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();moveBoundary(index,frame+(event.key==="ArrowLeft"?-1:1)*(event.shiftKey?10:1));};
    root.append(handle);
  });
  root.onclick = event => { const box=root.getBoundingClientRect(); manualSeek(Math.round((event.clientX-box.left)/box.width*(state.detail.episode_length-1))); };
  const tasks = $("subtask-list"); tasks.replaceChildren();
  state.session.subtasks.forEach((task,index) => { const li=document.createElement("li"); const skill=document.createElement("span"); skill.textContent=task.skill; const text=document.createElement("p"); text.textContent=task.text; li.append(skill,text); tasks.append(li); });
  updateFrameUI();
}

function beginBoundaryDrag(event,index,handle) {
  if (!canEdit()) return;
  event.preventDefault(); event.stopPropagation(); pauseAll(); $("play-button").textContent = "▶"; handle.setPointerCapture(event.pointerId);
  handle.onpointermove=pointer=>{
    const box=$("timeline").getBoundingClientRect();
    moveBoundary(index,frameFromPointer(pointer.clientX,box.left,box.width,state.detail.episode_length),false);
    handle.style.left=`${state.boundaries[index]/state.detail.episode_length*100}%`;
  };
  handle.onpointerup=()=>{handle.onpointermove=null;renderTimeline();renderDecision();rememberCurrentDraft();};
}

function moveBoundary(index,frame,rerender=true) {
  const low=index===0?1:state.boundaries[index-1]+1;
  const high=index===state.boundaries.length-1?state.detail.episode_length-1:state.boundaries[index+1]-1;
  state.boundaries[index]=Math.max(low,Math.min(high,frame)); manualSeek(state.boundaries[index]);
  if(rerender){renderTimeline();renderDecision();rememberCurrentDraft();}
}

function renderBoundaryList() {
  const root = $("boundary-list"); root.replaceChildren();
  state.boundaries.forEach((frame,index) => { const button=document.createElement("button"); button.textContent=`${index+1}: 帧 ${frame}`; button.onclick=()=>manualSeek(frame); root.append(button); });
}

function renderEvidence() {
  const root = $("evidence-content"); root.replaceChildren();
  const lines = [];
  if(state.detail.failure_category)lines.push(`失败类别：${state.detail.failure_category}`);
  state.detail.review_reasons.forEach(value=>lines.push(`复核原因：${value}`));
  state.detail.validation_issues.forEach(item=>lines.push(`约束问题 ${item.code}：${item.message}`));
  for (const attempt of state.detail.refine_attempts) lines.push(`精修边界 ${attempt.boundary_frame} · 置信度 ${attempt.confidence} · 可见线索：${attempt.visible_cues.join("；")}`);
  for (const attempt of state.detail.coarse_attempts) {
    lines.push(`粗标置信度 ${attempt.confidence} · 语义不确定：${attempt.semantic_uncertainty_codes.join("、")||"无"} · 精度说明：${attempt.boundary_precision_notes.join("；")||"无"}`);
    attempt.coarse_boundaries.forEach(boundary=>lines.push(`粗边界 ${boundary.from_subtask_index}→${boundary.to_subtask_index} @ ${boundary.estimated_frame}：${boundary.evidence}`));
  }
  if (!lines.length) root.textContent = "没有额外模型依据";
  else { const ul=document.createElement("ul"); lines.forEach(line=>{const li=document.createElement("li");li.textContent=line;ul.append(li)});root.append(ul); }
}

function renderDecision() {
  const highRisk = ["pending","failed","accepted"].includes(state.detail.status);
  const editable = canEdit();
  $("takeover-button").classList.toggle("hidden", !highRisk || state.takeover);
  $("takeover-button").textContent = state.detail.status === "accepted" ? "修正人工结果" : "人工接管";
  $("takeover-warning").classList.toggle("hidden", !highRisk);
  $("takeover-warning").textContent = highRisk ? `${statusLabel(state.detail.status)} episode 默认只读，必须显式确认后才可写入。` : "";
  const issues=renderValidation(); $("save-button").disabled = !editable || issues.length>0; $("start-subtask").disabled = !editable;
  $("add-boundary").disabled = !editable; $("remove-boundary").disabled = !editable;
}

function canEdit(){return Boolean(state.detail&&!state.loading&&!state.saving&&(state.detail.status==="needs_review"||(["pending","failed","accepted"].includes(state.detail.status)&&state.takeover)));}
function currentIssues(){return validateDraft({mode:state.detail.mode,subtaskCount:state.session.subtasks.length,length:state.detail.episode_length,minSegmentFrames:state.session.min_segment_frames},state.start,state.boundaries);}
function renderValidation(){const issues=currentIssues();const labels={start_subtask_range:"起始任务超出范围",complete_start_index:"complete 必须从任务 0 开始",complete_boundary_count:"complete 的边界数量不正确",dagger_suffix_length:"DAgger 后缀边界数量不正确",boundary_range:"边界必须位于 episode 内",boundary_order:"边界必须严格递增",segment_too_short:`区段短于 ${state.session.min_segment_frames} 帧`};setText("validation-message",issues.map(code=>labels[code]||code).join("；"));return issues;}
function rememberCurrentDraft(){if(state.detail)rememberDraft(state.drafts,state.detail.episode_index,{start:state.start,boundaries:state.boundaries,note:$("decision-note")?.value||"",takeover:state.takeover,status:state.detail.status,updated_at:state.detail.updated_at});}

function bindControls() {
  $("play-button").onclick = togglePlay; $("prev-frame").onclick=()=>step(-1); $("next-frame").onclick=()=>step(1);
  $("frame-slider").oninput = event => manualSeek(Number(event.target.value));
  $("start-subtask").onchange = event => { state.start=Number(event.target.value); renderTimeline();rememberCurrentDraft(); };
  $("decision-note").oninput=rememberCurrentDraft;
  $("add-boundary").onclick = () => { state.boundaries=insertBoundary(state.boundaries,state.frame,state.detail.episode_length);renderTimeline();renderDecision();rememberCurrentDraft(); };
  $("remove-boundary").onclick = () => { state.boundaries=removeNearestBoundary(state.boundaries,state.frame);renderTimeline();renderDecision();rememberCurrentDraft(); };
  $("takeover-button").onclick = () => { const action=state.detail.status==="accepted"?"修正已接受结果":"人工接管自动流程"; if(confirm(`确认${action} Episode ${state.detail.episode_index}？此操作会写入审计记录。`)){state.takeover=true;renderDecision();rememberCurrentDraft();} };
  $("save-button").onclick = saveDecision;
  document.addEventListener("keydown", event => {
    if (["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); togglePlay(); }
    if (event.key === "ArrowLeft") step(event.shiftKey?-10:-1);
    if (event.key === "ArrowRight") step(event.shiftKey?10:1);
    if (event.key.toLowerCase() === "b" && !$("add-boundary").disabled) $("add-boundary").click();
    if ((event.key === "Delete" || event.key === "Backspace") && !$("remove-boundary").disabled) $("remove-boundary").click();
  });
}

async function saveDecision() {
  if(state.loading||state.saving)return;
  setText("decision-message", "");
  if (!confirm(`确认将 Episode ${state.detail.episode_index} 的当前边界提交为 accepted？`)) return;
  const savedIndex=state.detail.episode_index; const saveGeneration=state.selectionGeneration;
  state.saving=true; renderEpisodes(); renderDecision();
  try {
    const payload=buildDecision(state.detail,state.start,state.boundaries,state.takeover,$("decision-note").value);
    const saved=await api(`/api/episodes/${savedIndex}/decision`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(!applyCommittedResponse(state,saved,savedIndex,saveGeneration))return;
    setText("status-badge",statusLabel(saved.status));$("status-badge").className=`status-badge ${saved.status}`;
    renderEvidence();renderTimeline();renderDecision();setText("decision-message","保存成功，workspace 已更新。");
    try {
      const [episodes,session]=await Promise.all([api("/api/episodes"),api("/api/session")]);
      state.episodes=episodes;state.session=session;renderFilters();renderEpisodes();renderDecision();
    } catch(refreshError) {
      setText("decision-message",`保存成功，但列表刷新失败：${refreshError.message}。请刷新页面。`);
    }
  } catch(error) { const issues=error.payload?.detail?.issues; setText("decision-message",issues?.length?`后端拒绝：${issues.join("、")}`:(error.message === "takeover_required" ? "请先点击人工接管。" : error.message)); }
  finally { state.saving=false; renderEpisodes(); if(state.detail)renderDecision(); }
}

if (typeof document !== "undefined") boot().catch(error => setText("workspace-summary", `加载失败：${error.message}`));
