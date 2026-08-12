const MAX_VIDEO_SECONDS = 10;

const state = {
  sessionId: getOrCreateSessionId(),
  task: null,
  selectedChoice: null,
  startedAt: null,
  submitting: false,
  reviewed: false,
  history: [],
};

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
const previousTaskButton = document.querySelector("#previous-task");
const nextTaskButton = document.querySelector("#next-task");
const responseClock = document.querySelector("#response-clock");
const sessionState = document.querySelector("#session-state");
const completePanel = document.querySelector("#complete-panel");
const restartButton = document.querySelector("#restart-review");

const modalityLabels = {
  text_to_video: "TEXT TO VIDEO",
  image_to_video: "IMAGE TO VIDEO",
  multi_reference: "MULTI REFERENCE",
  reference_material: "REFERENCE MATERIAL",
};

function getOrCreateSessionId() {
  const key = "human-signal-session";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const value = `browser-${crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`}`;
  window.localStorage.setItem(key, value);
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

function updateProgress(progress) {
  progressCurrent.textContent = progress?.current ?? "--";
  progressTotal.textContent = progress?.total ?? "--";
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
    reveal.classList.remove("is-ai", "is-real", "is-unknown");
    reveal.querySelector("strong").textContent = "";
  }
}

function showReveal(revealNode, source) {
  if (!source) return;
  const originType = source.origin_type || "unknown";
  revealNode.classList.remove("is-hidden");
  revealNode.classList.add(
    originType === "ai"
      ? "is-ai"
      : originType === "real"
        ? "is-real"
        : "is-unknown",
  );
  revealNode.querySelector("strong").textContent =
    source.label || "来源未标注";
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

  taskTitle.textContent = task.prompt
    ? "按条件比较两段表演"
    : "比较两段人物表演";
  taskId.textContent = `TASK ${task.task_id}`;
  modalityChip.textContent = modalityLabels[task.modality] || "VIDEO REVIEW";

  const hasPrompt = Boolean(String(task.prompt || "").trim());
  const hasReferences = Boolean(task.references?.length);
  contextCard.classList.toggle("is-hidden", !hasPrompt && !hasReferences);
  compareGrid.classList.toggle("has-context", hasPrompt || hasReferences);
  promptBlock.classList.toggle("is-hidden", !hasPrompt);
  contextBody.classList.toggle("reference-only", !hasPrompt && hasReferences);
  promptText.textContent = task.prompt || "本题未提供文字提示词。";

  renderReferences(task.references);
  resetMedia("video-a", task.candidates?.A);
  resetMedia("video-b", task.candidates?.B);
  resetReveal();
  if (options.reveal) {
    showReveal(revealA, options.reveal.A);
    showReveal(revealB, options.reveal.B);
  }

  choiceButtons.forEach((button) => {
    button.classList.toggle(
      "is-selected",
      button.dataset.choice === state.selectedChoice,
    );
    button.disabled = state.reviewed;
  });

  previousTaskButton.disabled = state.history.length === 0 || state.submitting;
  nextTaskButton.classList.toggle("is-hidden", !state.reviewed);
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
  previousTaskButton.disabled = true;
  nextTaskButton.classList.add("is-hidden");
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
      headers: { "X-Review-Session": state.sessionId },
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取任务");
    updateProgress(payload.progress);
    if (!payload.task) {
      renderComplete(payload.progress);
      return;
    }
    renderTask(payload.task);
  } catch (error) {
    sessionState.textContent = "CONNECTION ERROR";
    setMessage(error.message || "无法连接评测服务，请检查服务是否启动。", "error");
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
  try {
    const response = await fetch("/api/review/vote", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Review-Session": state.sessionId,
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
    state.history.push({
      task: JSON.parse(JSON.stringify(state.task)),
      choice,
      reveal,
    });
    showReveal(revealA, reveal.A);
    showReveal(revealB, reveal.B);
    responseClock.textContent = "已记录，结果揭示中...";
    await new Promise((resolve) => window.setTimeout(resolve, 950));
    state.submitting = false;
    await fetchNextTask();
  } catch (error) {
    state.submitting = false;
    choiceButtons.forEach((button) => {
      button.disabled = false;
    });
    setMessage(error.message || "投票记录失败，请重试。", "error");
  }
}

function showPreviousTask() {
  if (state.submitting || !state.history.length) return;
  const previous = state.history.pop();
  renderTask(previous.task, {
    reviewed: true,
    choice: previous.choice,
    reveal: previous.reveal,
  });
}

function continueToNextTask() {
  if (state.submitting || !state.reviewed) return;
  fetchNextTask();
}

function restartReview() {
  window.localStorage.setItem(
    "human-signal-session",
    `browser-${crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`}`,
  );
  window.location.reload();
}

choiceButtons.forEach((button) => {
  button.addEventListener("click", () => selectChoice(button.dataset.choice));
});
previousTaskButton.addEventListener("click", showPreviousTask);
nextTaskButton.addEventListener("click", continueToNextTask);
restartButton.addEventListener("click", restartReview);

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select, video")) return;
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
}, 1000);

fetchNextTask();
