const MAX_VIDEO_SECONDS = 10;

const state = {
  roundId: getOrCreateRoundId(),
  task: null,
  selectedChoice: null,
  startedAt: null,
  submitting: false,
  reviewed: false,
};

const qualityState = {
  roundId: getOrCreateRoundId("quality"),
  task: null,
  selectedRating: null,
  startedAt: null,
  submitting: false,
  reviewed: false,
  loaded: false,
};

let activeMode = "pairwise";

const modeTabs = [...document.querySelectorAll(".mode-tab")];
const pairwiseMain = document.querySelector("#pairwise-main");
const qualityMain = document.querySelector("#quality-main");
const progressCurrent = document.querySelector("#progress-current");
const progressTotal = document.querySelector("#progress-total");
const taskTitle = document.querySelector("#task-title");
const taskId = document.querySelector("#task-id");
const modalityChip = document.querySelector("#modality-chip");
const contextCard = document.querySelector("#context-card");
const contextBody = document.querySelector(".context-body");
const promptBlock = document.querySelector("#prompt-block");
const promptText = document.querySelector("#prompt-text");
const referenceStrip = document.querySelector("#reference-strip");
const revealA = document.querySelector("#reveal-a");
const revealB = document.querySelector("#reveal-b");
const messagePanel = document.querySelector("#message-panel");
const compareGrid = document.querySelector("#compare-grid");
const decisionPanel = document.querySelector(".decision-panel");
const choiceButtons = [...document.querySelectorAll(".choice-button")];
const responseClock = document.querySelector("#response-clock");
const sessionState = document.querySelector("#session-state");
const completePanel = document.querySelector("#complete-panel");
const decisionQuestion = document.querySelector("#decision-question");
const decisionHint = document.querySelector("#decision-hint");
const qualityProgressCurrent = document.querySelector("#quality-progress-current");
const qualityProgressTotal = document.querySelector("#quality-progress-total");
const qualityTaskTitle = document.querySelector("#quality-task-title");
const qualityTaskId = document.querySelector("#quality-task-id");
const qualityModalityChip = document.querySelector("#quality-modality-chip");
const qualityMessagePanel = document.querySelector("#quality-message-panel");
const qualityVideo = document.querySelector("#quality-video");
const qualityVideoFrame = document.querySelector(".quality-video-frame");
const qualityQuestion = document.querySelector("#quality-question");
const qualityHint = document.querySelector("#quality-hint");
const qualityRatingButtons = [
  ...document.querySelectorAll(".quality-rating-button"),
];
const qualityResponseClock = document.querySelector("#quality-response-clock");
const qualityCompletePanel = document.querySelector("#quality-complete-panel");
const qualityDecisionPanel = document.querySelector(".quality-decision-panel");

const modalityLabels = {
  text_to_video: "TEXT TO VIDEO",
  image_to_video: "IMAGE TO VIDEO",
  multi_reference: "MULTI REFERENCE",
  reference_material: "REFERENCE MATERIAL",
};

function getOrCreateRoundId(suffix = "pairwise") {
  const key = `human-signal-round-${suffix}`;
  const existing = window.sessionStorage.getItem(key);
  if (existing) return existing;
  const value = `round-${crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`}`;
  window.sessionStorage.setItem(key, value);
  return value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const remainder = String(total % 60).padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function setMessage(message, tone = "normal") {
  if (!message) {
    messagePanel.classList.add("is-hidden");
    messagePanel.textContent = "";
    return;
  }
  messagePanel.textContent = message;
  messagePanel.dataset.tone = tone;
  messagePanel.classList.remove("is-hidden");
}

function setQualityMessage(message, tone = "normal") {
  if (!message) {
    qualityMessagePanel.classList.add("is-hidden");
    qualityMessagePanel.textContent = "";
    return;
  }
  qualityMessagePanel.textContent = message;
  qualityMessagePanel.dataset.tone = tone;
  qualityMessagePanel.classList.remove("is-hidden");
}

function updateProgress(progress) {
  progressCurrent.textContent = progress?.current ?? "--";
  progressTotal.textContent = progress?.total ?? "--";
}

function updateQualityProgress(progress) {
  qualityProgressCurrent.textContent = progress?.current ?? "--";
  qualityProgressTotal.textContent = progress?.total ?? "--";
}

function resetMedia(videoId, asset) {
  const video = document.querySelector(`#${videoId}`);
  const frame = video.closest(".video-frame");
  const empty = frame.querySelector(".video-empty");
  if (video._reviewCleanup) video._reviewCleanup();

  video.pause();
  video.autoplay = true;
  video.muted = true;
  video.removeAttribute("src");
  video.removeAttribute("poster");
  video.load();
  frame.classList.remove("has-media", "media-error");
  empty.classList.remove("is-hidden");

  if (!asset?.url) return;

  video.src = asset.url;
  if (asset.poster) video.poster = asset.poster;

  const enforceLimit = () => {
    if (video.currentTime >= MAX_VIDEO_SECONDS) {
      video.currentTime = MAX_VIDEO_SECONDS;
      video.pause();
    }
  };
  const onLoadedMetadata = () => {
    video.currentTime = 0;
    video.play().catch(() => {});
  };
  const onLoadedData = () => {
    frame.classList.add("has-media");
    empty.classList.add("is-hidden");
  };
  const onError = () => {
    frame.classList.add("media-error");
    empty.classList.remove("is-hidden");
    empty.querySelector("strong").textContent = "视频无法播放";
    empty.querySelector("small").textContent =
      "当前素材编码或文件不可用，请联系管理员";
  };

  video.addEventListener("loadedmetadata", onLoadedMetadata);
  video.addEventListener("loadeddata", onLoadedData, { once: true });
  video.addEventListener("timeupdate", enforceLimit);
  video.addEventListener("seeking", enforceLimit);
  video.addEventListener("error", onError, { once: true });
  video._reviewCleanup = () => {
    video.removeEventListener("loadedmetadata", onLoadedMetadata);
    video.removeEventListener("loadeddata", onLoadedData);
    video.removeEventListener("timeupdate", enforceLimit);
    video.removeEventListener("seeking", enforceLimit);
    video.removeEventListener("error", onError);
  };
  video.load();
}

function resetReveal() {
  for (const reveal of [revealA, revealB]) {
    reveal.classList.add("is-hidden");
    reveal.classList.remove("is-ai", "is-real", "is-model", "is-unknown");
    reveal.querySelector("strong").textContent = "";
  }
  for (const label of document.querySelectorAll(".candidate-label")) {
    label.textContent = "LABEL HIDDEN";
    label.classList.remove("is-revealed");
  }
}

function setCandidateLabel(candidateLabel, source) {
  const label = document.querySelector(
    `.candidate-label[data-candidate-label="${candidateLabel}"]`,
  );
  if (!label || !source) return;
  label.textContent = source.label || "LABEL REVEALED";
  label.classList.add("is-revealed");
}

function showReveal(revealNode, source, candidateLabel) {
  if (!source) return;
  const revealMode = source.reveal_mode || "origin";
  const originType = source.origin_type || "unknown";
  revealNode.classList.remove("is-hidden");
  revealNode.classList.add(
    revealMode === "model"
      ? "is-model"
      : originType === "ai"
        ? "is-ai"
        : originType === "real"
          ? "is-real"
          : "is-unknown",
  );
  revealNode.querySelector("span").textContent =
    revealMode === "model" ? "REVEALED MODEL" : "REVEALED SOURCE";
  revealNode.querySelector("strong").textContent =
    source.label || "来源未标注";
  setCandidateLabel(candidateLabel, source);
}

function renderReferences(references) {
  if (!references?.length) {
    referenceStrip.innerHTML = "";
    return;
  }

  referenceStrip.innerHTML = references
    .map((reference, index) => {
      const label = escapeHtml(
        reference.label ||
          (reference.type === "video"
            ? `参考视频 ${index + 1}`
            : `参考内容 ${index + 1}`),
      );
      let media;
      if (reference.type === "video") {
        media = `<video src="${escapeHtml(reference.url)}" ${
          reference.poster ? `poster="${escapeHtml(reference.poster)}"` : ""
        } controls muted playsinline preload="metadata"></video>`;
      } else if (reference.type === "audio") {
        media = `<audio src="${escapeHtml(reference.url)}" controls preload="metadata"></audio>`;
      } else {
        media = `<img src="${escapeHtml(reference.url)}" alt="${label}" />`;
      }
      return `
        <figure class="reference-item">
          <div class="reference-media">${media}</div>
          <figcaption><span>${escapeHtml(reference.type)}</span>${label}</figcaption>
        </figure>
      `;
    })
    .join("");
}

function renderTask(task, options = {}) {
  state.task = task;
  state.reviewed = Boolean(options.reviewed);
  state.selectedChoice = options.choice || null;
  state.startedAt = state.reviewed ? null : performance.now();

  const taskType = task.task_type || "ai_real_anchor";
  const hasContext = Boolean(task.show_context);
  taskTitle.textContent =
    taskType === "model_comparison"
      ? "同一条件下比较模型表现"
      : task.prompt
        ? "按条件比较两段表演"
        : "比较两段人物表演";
  taskId.textContent = `TASK ${task.task_id}`;
  modalityChip.textContent = modalityLabels[task.modality] || "VIDEO REVIEW";

  const hasPrompt = hasContext && Boolean(String(task.prompt || "").trim());
  const hasReferences = hasContext && Boolean(task.references?.length);
  contextCard.classList.toggle("is-hidden", !hasPrompt && !hasReferences);
  compareGrid.classList.toggle("has-context", hasPrompt || hasReferences);
  promptBlock.classList.toggle("is-hidden", !hasPrompt);
  contextBody.classList.toggle("reference-only", !hasPrompt && hasReferences);
  promptText.textContent = task.prompt || "本题未提供文字提示词。";
  decisionQuestion.textContent =
    task.question || "哪个视频中的人物表演更像真人？";
  decisionHint.textContent =
    taskType === "model_comparison"
      ? "只比较人物表演质量；请不要根据画面风格、文件名或品牌偏好作答。"
      : "如果差异不明显，或视频无法判断，请选择第三项。";

  renderReferences(task.references);
  resetMedia("video-a", task.candidates?.A);
  resetMedia("video-b", task.candidates?.B);
  resetReveal();
  if (options.reveal) {
    showReveal(revealA, options.reveal.A, "A");
    showReveal(revealB, options.reveal.B, "B");
  }

  choiceButtons.forEach((button) => {
    button.classList.toggle(
      "is-selected",
      button.dataset.choice === state.selectedChoice,
    );
    button.disabled = state.reviewed;
  });

  responseClock.textContent = state.reviewed
    ? "已完成本题"
    : "观看时间 00:00";
  decisionPanel.classList.remove("is-disabled");
  completePanel.classList.add("is-hidden");
  compareGrid.classList.remove("is-hidden");
  setMessage("");
  sessionState.textContent = state.reviewed
    ? "REVIEWED / BACK"
    : "SESSION ACTIVE";
}

function renderComplete(progress) {
  updateProgress(progress);
  state.task = null;
  state.reviewed = false;
  compareGrid.classList.add("is-hidden");
  decisionPanel.classList.add("is-disabled");
  completePanel.classList.remove("is-hidden");
  taskTitle.textContent = "本轮评测已完成";
  taskId.textContent = "NO PENDING TASK";
  modalityChip.textContent = "COMPLETE";
  sessionState.textContent = "ROUND COMPLETE";
  setMessage("");
}

async function fetchNextTask() {
  sessionState.textContent = "LOADING TASK";
  setMessage("正在读取下一道评测任务...");
  const currentId = state.task?.task_id;
  const query = currentId ? `?task_id=${encodeURIComponent(currentId)}` : "";
  try {
    const response = await fetch(`/api/review/next${query}`, {
      headers: {
        "X-Review-Round": state.roundId,
      },
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取任务");
    updateProgress(payload.progress);
    if (!payload.task) {
      renderComplete(payload.progress);
      return false;
    }
    renderTask(payload.task);
    return true;
  } catch (error) {
    sessionState.textContent = "CONNECTION ERROR";
    setMessage(error.message || "无法连接评测服务，请检查服务是否启动。", "error");
    return null;
  }
}

async function selectChoice(choice) {
  if (!state.task || state.submitting || state.reviewed) return;
  state.selectedChoice = choice;
  state.submitting = true;
  choiceButtons.forEach((button) => {
    button.classList.toggle("is-selected", button.dataset.choice === choice);
    button.disabled = true;
  });

  const responseMs = Math.round(performance.now() - state.startedAt);
  document.querySelectorAll(".video-frame video").forEach((video) => {
    video.pause();
  });
  try {
    const response = await fetch("/api/review/vote", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Review-Round": state.roundId,
      },
      body: JSON.stringify({
        task_id: state.task.task_id,
        choice,
        response_ms: responseMs,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "投票记录失败");

    updateProgress(payload.progress);
    const reveal = payload.progress?.reveal || {};
    showReveal(revealB, reveal.B, "B");
    showReveal(revealA, reveal.A, "A");
    responseClock.textContent = "已记录，约 2 秒后进入下一题";
    state.reviewed = true;
    choiceButtons.forEach((button) => {
      button.disabled = true;
    });
    sessionState.textContent = "RESULT REVEALED";
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
    state.submitting = false;
    state.reviewed = false;
    const nextStatus = await fetchNextTask();
    if (nextStatus === null) {
      state.reviewed = true;
      responseClock.textContent = "本题已记录，请刷新页面继续下一题";
    }
  } catch (error) {
    state.submitting = false;
    choiceButtons.forEach((button) => {
      button.disabled = false;
    });
    setMessage(error.message || "投票记录失败，请重试。", "error");
  }
}

function resetQualityMedia(asset) {
  if (qualityVideo._reviewCleanup) qualityVideo._reviewCleanup();

  const empty = qualityVideoFrame.querySelector(".video-empty");
  qualityVideo.pause();
  qualityVideo.autoplay = true;
  qualityVideo.muted = true;
  qualityVideo.removeAttribute("src");
  qualityVideo.removeAttribute("poster");
  qualityVideo.load();
  qualityVideoFrame.classList.remove("has-media", "media-error");
  empty.classList.remove("is-hidden");
  empty.querySelector("strong").textContent = "视频加载中";
  empty.querySelector("small").textContent =
    "如果长时间未显示，将提示具体错误";

  if (!asset?.url) return;

  qualityVideo.src = asset.url;
  const onLoadedMetadata = () => {
    qualityVideo.currentTime = 0;
    qualityVideo.play().catch(() => {});
  };
  const onLoadedData = () => {
    qualityVideoFrame.classList.add("has-media");
    empty.classList.add("is-hidden");
  };
  const onError = () => {
    qualityVideoFrame.classList.add("media-error");
    empty.classList.remove("is-hidden");
    empty.querySelector("strong").textContent = "视频无法播放";
    empty.querySelector("small").textContent =
      "当前素材编码或文件不可用，请联系管理员";
  };

  qualityVideo.addEventListener("loadedmetadata", onLoadedMetadata);
  qualityVideo.addEventListener("loadeddata", onLoadedData, { once: true });
  qualityVideo.addEventListener("error", onError, { once: true });
  qualityVideo._reviewCleanup = () => {
    qualityVideo.removeEventListener("loadedmetadata", onLoadedMetadata);
    qualityVideo.removeEventListener("loadeddata", onLoadedData);
    qualityVideo.removeEventListener("error", onError);
  };
  qualityVideo.load();
}

function renderQualityTask(task) {
  qualityState.task = task;
  qualityState.reviewed = false;
  qualityState.selectedRating = null;
  qualityState.startedAt = performance.now();

  qualityTaskTitle.textContent = "正在观看一段 AI 视频";
  qualityTaskId.textContent = "SINGLE VIDEO";
  qualityModalityChip.textContent = "AI QUALITY";
  qualityQuestion.textContent =
    task.question || "这段视频属于哪个质量档次？";
  qualityHint.textContent = "以人物整体表现为准，不参考文件名、来源或程序分数。";
  resetQualityMedia(task.video);
  qualityRatingButtons.forEach((button) => {
    button.classList.remove("is-selected");
    button.disabled = false;
  });
  qualityResponseClock.textContent = "观看时间 00:00";
  qualityDecisionPanel.classList.remove("is-disabled");
  qualityCompletePanel.classList.add("is-hidden");
  setQualityMessage("");
  sessionState.textContent = "AI QUALITY ACTIVE";
}

function renderQualityComplete(progress) {
  updateQualityProgress(progress);
  qualityState.task = null;
  qualityState.reviewed = false;
  qualityDecisionPanel.classList.add("is-disabled");
  qualityCompletePanel.classList.remove("is-hidden");
  qualityTaskTitle.textContent = "AI 质量评测已完成";
  qualityTaskId.textContent = "NO PENDING VIDEO";
  qualityModalityChip.textContent = "COMPLETE";
  sessionState.textContent = "AI QUALITY COMPLETE";
  setQualityMessage("");
}

async function fetchNextQualityTask() {
  sessionState.textContent = "LOADING AI QUALITY";
  setQualityMessage("正在读取下一条 AI 视频...");
  const currentId = qualityState.task?.task_id;
  const query = currentId ? `?task_id=${encodeURIComponent(currentId)}` : "";
  try {
    const response = await fetch(`/api/quality/next${query}`, {
      headers: {
        "X-Review-Round": qualityState.roundId,
      },
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取 AI 质量任务");
    qualityState.loaded = true;
    updateQualityProgress(payload.progress);
    if (!payload.task) {
      renderQualityComplete(payload.progress);
      return false;
    }
    renderQualityTask(payload.task);
    return true;
  } catch (error) {
    qualityState.loaded = false;
    sessionState.textContent = "QUALITY CONNECTION ERROR";
    setQualityMessage(
      error.message || "无法连接 AI 质量评测服务，请检查数据集是否已构建。",
      "error",
    );
    return null;
  }
}

async function selectQualityRating(rating) {
  if (
    !qualityState.task ||
    qualityState.submitting ||
    qualityState.reviewed
  ) {
    return;
  }
  qualityState.selectedRating = rating;
  qualityState.submitting = true;
  qualityRatingButtons.forEach((button) => {
    button.classList.toggle("is-selected", button.dataset.rating === rating);
    button.disabled = true;
  });
  const responseMs = Math.round(
    performance.now() - qualityState.startedAt,
  );
  qualityVideo.pause();
  try {
    const response = await fetch("/api/quality/rate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Review-Round": qualityState.roundId,
      },
      body: JSON.stringify({
        task_id: qualityState.task.task_id,
        rating,
        response_ms: responseMs,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "质量评分记录失败");
    updateQualityProgress(payload.progress);
    qualityState.reviewed = true;
    qualityResponseClock.textContent = "已记录，正在进入下一条视频";
    sessionState.textContent = "AI QUALITY RECORDED";
    await new Promise((resolve) => window.setTimeout(resolve, 650));
    qualityState.submitting = false;
    qualityState.reviewed = false;
    const nextStatus = await fetchNextQualityTask();
    if (nextStatus === null) {
      qualityState.reviewed = true;
      qualityResponseClock.textContent = "本题已记录，请刷新页面继续";
    }
  } catch (error) {
    qualityState.submitting = false;
    qualityRatingButtons.forEach((button) => {
      button.disabled = false;
    });
    setQualityMessage(error.message || "质量评分记录失败，请重试。", "error");
  }
}

function setMode(mode) {
  activeMode = mode === "quality" ? "quality" : "pairwise";
  const qualityVisible = activeMode === "quality";
  pairwiseMain.classList.toggle("is-hidden", qualityVisible);
  qualityMain.classList.toggle("is-hidden", !qualityVisible);
  modeTabs.forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.mode === activeMode);
  });
  if (qualityVisible) {
    if (!qualityState.loaded) fetchNextQualityTask();
    return;
  }
  sessionState.textContent = state.task ? "SESSION ACTIVE" : "SESSION READY";
}

qualityRatingButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectQualityRating(button.dataset.rating);
  });
});

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

choiceButtons.forEach((button) => {
  button.addEventListener("click", () => selectChoice(button.dataset.choice));
});

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select, video")) return;
  if (activeMode === "quality") {
    const ratingByKey = { "1": "upper", "2": "middle", "3": "lower" };
    const rating = ratingByKey[event.key];
    if (rating) {
      event.preventDefault();
      selectQualityRating(rating);
    }
    return;
  }
  const choiceByKey = { a: "A", b: "B", c: "tie_or_unrateable" };
  const choice = choiceByKey[event.key.toLowerCase()];
  if (choice) {
    event.preventDefault();
    selectChoice(choice);
  }
});

window.setInterval(() => {
  if (state.startedAt && state.task && !state.submitting && !state.reviewed) {
    responseClock.textContent = `观看时间 ${formatDuration(
      (performance.now() - state.startedAt) / 1000,
    )}`;
  }
  if (
    qualityState.startedAt &&
    qualityState.task &&
    !qualityState.submitting &&
    !qualityState.reviewed &&
    activeMode === "quality"
  ) {
    qualityResponseClock.textContent = `观看时间 ${formatDuration(
      (performance.now() - qualityState.startedAt) / 1000,
    )}`;
  }
}, 1000);

fetchNextTask();
