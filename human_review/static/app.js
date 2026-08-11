const state = {
  sessionId: getOrCreateSessionId(),
  task: null,
  selectedChoice: null,
  startedAt: null,
  submitting: false,
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
const submitVoteButton = document.querySelector("#submit-vote");
const responseClock = document.querySelector("#response-clock");
const sessionState = document.querySelector("#session-state");
const completePanel = document.querySelector("#complete-panel");
const restartButton = document.querySelector("#restart-review");

const modalityLabels = {
  text_to_video: "TEXT TO VIDEO",
  image_to_video: "IMAGE TO VIDEO",
  multi_reference: "MULTI REFERENCE",
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
  video.pause();
  video.removeAttribute("src");
  video.removeAttribute("poster");
  video.load();
  frame.classList.remove("has-media", "media-error");
  empty.classList.remove("is-hidden");

  if (!asset?.url) return;
  video.src = asset.url;
  if (asset.poster) video.poster = asset.poster;
  video.addEventListener(
    "loadeddata",
    () => {
      frame.classList.add("has-media");
      empty.classList.add("is-hidden");
    },
    { once: true },
  );
  video.addEventListener(
    "error",
    () => {
      frame.classList.add("media-error");
      empty.classList.remove("is-hidden");
    },
    { once: true },
  );
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
    referenceStrip.innerHTML =
      '<div class="reference-empty">此题没有额外参考内容</div>';
    return;
  }

  referenceStrip.innerHTML = references
    .map((reference, index) => {
      const label = escapeHtml(
        reference.label ||
          (reference.type === "video" ? `参考视频 ${index + 1}` : `参考图 ${index + 1}`),
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

function renderTask(task) {
  state.task = task;
  state.selectedChoice = null;
  state.startedAt = performance.now();
  taskTitle.textContent = task.prompt
    ? "按条件比较两段表演"
    : "比较两段人物表演";
  taskId.textContent = `TASK ${task.task_id}`;
  modalityChip.textContent = modalityLabels[task.modality] || "VIDEO REVIEW";
  const hasPrompt = Boolean(String(task.prompt || "").trim());
  const hasReferences = Boolean(task.references?.length);
  contextCard.classList.toggle("is-hidden", !hasPrompt && !hasReferences);
  promptBlock.classList.toggle("is-hidden", !hasPrompt);
  contextBody.classList.toggle("reference-only", !hasPrompt && hasReferences);
  promptText.textContent = task.prompt || "";
  renderReferences(task.references);
  resetMedia("video-a", task.candidates?.A);
  resetMedia("video-b", task.candidates?.B);
  resetReveal();
  choiceButtons.forEach((button) => {
    button.classList.remove("is-selected");
    button.disabled = false;
  });
  submitVoteButton.disabled = true;
  submitVoteButton.classList.remove("is-loading");
  responseClock.textContent = "观看时间 00:00";
  decisionPanel.classList.remove("is-disabled");
  completePanel.classList.add("is-hidden");
  compareGrid.classList.remove("is-hidden");
  setMessage("");
  sessionState.textContent = "SESSION ACTIVE";
}

function renderComplete(progress) {
  updateProgress(progress);
  state.task = null;
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

function selectChoice(choice) {
  if (!state.task || state.submitting) return;
  state.selectedChoice = choice;
  choiceButtons.forEach((button) => {
    button.classList.toggle("is-selected", button.dataset.choice === choice);
  });
  submitVoteButton.disabled = false;
}

async function submitVote() {
  if (!state.task || !state.selectedChoice || state.submitting) return;
  state.submitting = true;
  submitVoteButton.disabled = true;
  submitVoteButton.classList.add("is-loading");
  submitVoteButton.querySelector("span").textContent = "正在记录...";
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
        choice: state.selectedChoice,
        response_ms: responseMs,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "投票记录失败");
    updateProgress(payload.progress);
    showReveal(revealA, payload.progress?.reveal?.A);
    showReveal(revealB, payload.progress?.reveal?.B);
    submitVoteButton.querySelector("span").textContent = "已记录，结果揭示中...";
    submitVoteButton.classList.remove("is-loading");
    await new Promise((resolve) => window.setTimeout(resolve, 1700));
    state.submitting = false;
    submitVoteButton.querySelector("span").textContent = "提交并进入下一题";
    await fetchNextTask();
  } catch (error) {
    state.submitting = false;
    submitVoteButton.disabled = false;
    submitVoteButton.classList.remove("is-loading");
    submitVoteButton.querySelector("span").textContent = "提交并进入下一题";
    setMessage(error.message || "投票记录失败，请重试。", "error");
  }
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
submitVoteButton.addEventListener("click", submitVote);
restartButton.addEventListener("click", restartReview);

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select, video")) return;
  const choiceByKey = { a: "A", b: "B", c: "tie_or_unrateable" };
  const choice = choiceByKey[event.key.toLowerCase()];
  if (choice) {
    event.preventDefault();
    selectChoice(choice);
  }
  if (event.key === "Enter" && state.selectedChoice) {
    event.preventDefault();
    submitVote();
  }
});

window.setInterval(() => {
  if (state.startedAt && state.task && !state.submitting) {
    responseClock.textContent = `观看时间 ${formatDuration(
      (performance.now() - state.startedAt) / 1000,
    )}`;
  }
}, 1000);

fetchNextTask();
