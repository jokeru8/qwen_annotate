"use strict";
document.querySelectorAll(".decision-form").forEach((form) => {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const article = form.closest("article");
    const startText = form.elements.start_subtask_index.value.trim();
    const boundaryText = form.elements.boundaries.value.trim();
    if (!/^(0|[1-9][0-9]*)$/.test(startText)) { window.alert("Start must be a non-negative integer."); return; }
    const parts = boundaryText === "" ? [] : boundaryText.split(",").map((item) => item.trim());
    if (parts.some((item) => !/^(0|[1-9][0-9]*)$/.test(item))) { window.alert("Boundaries must be comma-separated non-negative integers."); return; }
    const decision = {
      episode_index: Number(article.dataset.episode),
      source_fingerprint: article.dataset.fingerprint,
      run_fingerprint: article.dataset.runFingerprint,
      mode: article.dataset.mode,
      start_subtask_index: Number(startText),
      boundaries: parts.map(Number),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(decision)], {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = `decision_episode_${String(decision.episode_index).padStart(6, "0")}.json`;
    link.click();
    URL.revokeObjectURL(url);
  });
});
