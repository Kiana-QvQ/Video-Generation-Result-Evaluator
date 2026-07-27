const form = document.querySelector("#evaluation-form");
const evaluateButton = document.querySelector("#evaluate-button");
const formNote = document.querySelector("#form-note");
const modelList = document.querySelector("#model-list");
const modelTime = document.querySelector("#model-time");
const emptyReport = document.querySelector("#empty-report");
const reportContent = document.querySelector("#report-content");
const reportMode = document.querySelector("#report-mode");
const overallScore = document.querySelector("#overall-score");
const scoreCaption = document.querySelector("#score-caption");
const coverageValue = document.querySelector("#coverage-value");
const coverageRing = document.querySelector("#coverage-ring");
const categoryList = document.querySelector("#category-list");
const evidenceGrid = document.querySelector("#evidence-grid");
const downloadRow = document.querySelector("#download-row");
const processProgress = document.querySelector("#process-progress");
const progressLabel = document.querySelector("#progress-label");
const progressTime = document.querySelector("#progress-time");
const progressBar = document.querySelector("#progress-bar");
const progressSteps = [...document.querySelectorAll("[data-progress-step]")];
const queueSummary = document.querySelector("#queue-summary");
const queueActive = document.querySelector("#queue-active");
const activeJobName = document.querySelector("#active-job-name");
const activeJobStage = document.querySelector("#active-job-stage");
const activeJobStatus = document.querySelector("#active-job-status");
const activeJobProgress = document.querySelector("#active-job-progress");
const queueList = document.querySelector("#queue-list");
const queueEmpty = document.querySelector("#queue-empty");
const refreshQueueButton = document.querySelector("#refresh-queue");
const previewUrls = new Map();

const categoryOrder = [
  ["identity", "1. 角色一致性", 35],
  ["texture", "2. 质感和细节", 15],
  ["expression", "3. 表情准确", 15],
  ["temporal", "4. 时间稳定性", 25],
  ["aesthetics", "5. 美学质量", 10],
];

const modeLabels = {
  full_reference: "FULL REFERENCE",
  reference_material: "REFERENCE MATERIAL",
  prompt_only: "PROMPT ONLY",
  result_only: "RESULT ONLY",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function describeApiError(detail, status) {
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter(Boolean);
    if (messages.length) return messages.join("；");
  }
  if (typeof detail === "string" && detail) return detail;
  return `评估失败（HTTP ${status}）`;
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toFixed(digits);
}

function normalizeScore(value) {
  if (value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : null;
}

function radarPoints(values, centerX, centerY, radius) {
  return values
    .map((value, index) => {
      const angle = -Math.PI / 2 + (Math.PI * 2 * index) / values.length;
      const distance = radius * (value ?? 0);
      return `${(centerX + Math.cos(angle) * distance).toFixed(2)},${(
        centerY + Math.sin(angle) * distance
      ).toFixed(2)}`;
    })
    .join(" ");
}

function renderRadar(result) {
  const values = categoryOrder.map(([key]) =>
    normalizeScore(result.categories?.[key]?.metrics?.score_0_1),
  );
  const labels = ["IDENTITY", "TEXTURE", "EXPRESSION", "TEMPORAL", "AESTHETICS"];
  const width = 350;
  const height = 190;
  const centerX = 175;
  const centerY = 98;
  const radius = 67;
  const levels = [0.25, 0.5, 0.75, 1];
  const grid = levels
    .map(
      (level) =>
        `<polygon class="radar-grid" points="${radarPoints(
          Array(5).fill(level),
          centerX,
          centerY,
          radius,
        )}" />`,
    )
    .join("");
  const axes = labels
    .map((label, index) => {
      const angle = -Math.PI / 2 + (Math.PI * 2 * index) / labels.length;
      const endX = centerX + Math.cos(angle) * radius;
      const endY = centerY + Math.sin(angle) * radius;
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 20);
      return `
        <line class="radar-axis" x1="${centerX}" y1="${centerY}" x2="${endX.toFixed(2)}" y2="${endY.toFixed(2)}" />
        <text class="radar-label" x="${labelX.toFixed(2)}" y="${labelY.toFixed(2)}">${label}</text>
      `;
    })
    .join("");
  const area = radarPoints(values, centerX, centerY, radius);
  const points = values
    .map((value, index) => {
      const angle = -Math.PI / 2 + (Math.PI * 2 * index) / values.length;
      const distance = radius * (value ?? 0);
      return `<circle class="radar-point" cx="${(centerX + Math.cos(angle) * distance).toFixed(2)}" cy="${(
        centerY + Math.sin(angle) * distance
      ).toFixed(2)}" r="3.4" />`;
    })
    .join("");
  document.querySelector("#radar-chart").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="五类评分雷达图">
      ${grid}
      ${axes}
      <polygon class="radar-area" points="${area}" />
      ${points}
    </svg>
  `;
}

function statusLabel(status) {
  return status === "available" ? "ready" : status === "manual" ? "manual" : status;
}

function attachFileInput(id) {
  const input = document.querySelector(`#${id}`);
  if (!input) return;
  const name = document.querySelector(`[data-file-name="${id}"]`);
  const zone = input.closest("label");
  const preview = document.querySelector(`[data-preview="${id}"]`);
  const fallback = document.querySelector(`[data-preview-fallback="${id}"]`);
  const imageList = document.querySelector(`[data-image-list="${id}"]`);
  input.addEventListener("change", () => {
    const files = [...(input.files ?? [])];
    const file = files[0];
    if (!file) return;
    name.textContent = input.multiple
      ? `${files.length} reference image${files.length === 1 ? "" : "s"}`
      : file.name;
    zone.classList.add("is-loaded");
    if (imageList) {
      imageList.innerHTML = "";
      files.forEach((imageFile) => {
        const image = document.createElement("img");
        image.alt = imageFile.name;
        image.src = URL.createObjectURL(imageFile);
        previewUrls.set(
          `${id}:${imageFile.name}`,
          image.src,
        );
        imageList.appendChild(image);
      });
      return;
    }
    if (preview && file.type.startsWith("video/")) {
      const previousUrl = previewUrls.get(id);
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      const objectUrl = URL.createObjectURL(file);
      previewUrls.set(id, objectUrl);
      zone.classList.remove("video-preview-failed");
      if (fallback) fallback.classList.remove("is-visible");
      preview.src = objectUrl;
      preview.load();
      preview.addEventListener(
        "error",
        () => {
          zone.classList.add("video-preview-failed");
          if (fallback) fallback.classList.add("is-visible");
        },
        { once: true },
      );
      preview.addEventListener(
        "loadedmetadata",
        () => {
          const seconds = Number.isFinite(preview.duration)
            ? ` · ${preview.duration.toFixed(1)}s`
            : "";
          name.textContent = `${file.name}${seconds}`;
        },
        { once: true },
      );
      window.setTimeout(() => {
        if (preview.readyState === 0) {
          zone.classList.add("video-preview-failed");
          if (fallback) fallback.classList.add("is-visible");
        }
      }, 1200);
    }
  });
}

["result-video", "gt-video", "reference-images", "reference-video"].forEach(attachFileInput);

window.addEventListener("beforeunload", () => {
  previewUrls.forEach((url) => {
    if (Array.isArray(url)) {
      url.forEach((item) => URL.revokeObjectURL(item));
    } else {
      URL.revokeObjectURL(url);
    }
  });
  previewUrls.clear();
});

function renderModels(payload) {
  modelTime.textContent = `cache checked / ${payload.generated_at?.slice(11, 19) ?? "local"}`;
  modelList.innerHTML = payload.models
    .map((model) => {
      const label =
        model.status === "ready"
          ? "READY"
          : model.status === "optional"
            ? "OPTIONAL"
            : "OFFLINE";
      return `
        <div class="model-chip ${escapeHtml(model.status)}" title="${escapeHtml(model.note)}">
          <span class="status-dot"></span>
          <span>${escapeHtml(model.name)}</span>
          <small>${label}</small>
        </div>
      `;
    })
    .join("");
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("model endpoint unavailable");
    renderModels(await response.json());
  } catch (error) {
    modelTime.textContent = "model cache unavailable";
    modelList.innerHTML =
      '<div class="model-chip optional"><span class="status-dot"></span><span>LOCAL BACKENDS</span><small>CHECK FAILED</small></div>';
  }
}

function setFormNote(message, tone = "normal") {
  formNote.innerHTML = `<span class="note-pin"></span>${escapeHtml(message)}`;
  formNote.dataset.tone = tone;
}

function renderCategories(result) {
  categoryList.innerHTML = categoryOrder
    .map(([key, label, weight]) => {
      const category = result.categories?.[key] ?? {};
      const score = normalizeScore(category.metrics?.score_0_1);
      const scoreText = score === null ? "—" : `${Math.round(score * 100)}`;
      const status = statusLabel(category.status ?? "unavailable");
      return `
        <div class="category-row">
          <div>
            <div class="category-topline">
              <span>${label}</span>
              <small>${scoreText}${score === null ? "" : "/100"} · ${weight}%</small>
            </div>
          </div>
          <span class="category-status ${escapeHtml(status)}">${escapeHtml(status)}</span>
          <div class="category-track">
            <div class="category-fill" style="width: ${score === null ? 0 : score * 100}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderEvidence(result) {
  const categories = result.categories ?? {};
  const texture = categories.texture?.metrics ?? {};
  const identity = categories.identity?.metrics ?? {};
  const expression = categories.expression?.metrics ?? {};
  const temporal = categories.temporal?.metrics ?? {};
  const aesthetics = categories.aesthetics?.metrics ?? {};
  const fullReference = categories.texture?.mode === "full_reference";
  const evidence = [
    ["IDENTITY / MEAN", formatNumber(identity.mean_similarity), "ArcFace / proxy"],
    [fullReference ? "PSNR / dB" : "TEXTURE / SCORE", fullReference ? formatNumber(texture.psnr_db, 2) : formatNumber(texture.score_0_1), fullReference ? "GT reference" : "no GT"],
    [fullReference ? "SSIM" : "MANIQA", fullReference ? formatNumber(texture.ssim, 4) : formatNumber(texture.maniqa), fullReference ? "GT reference" : "optional IQA"],
    [fullReference ? "LPIPS" : "MUSIQ", fullReference ? formatNumber(texture.lpips, 4) : formatNumber(texture.musiq), fullReference ? "lower is better" : "optional IQA"],
    ["TEXT / VIDEO", formatNumber(expression.text_video_alignment), "CLIP baseline"],
    ["TEMPORAL / STABILITY", formatNumber(temporal.stability_score_0_1), "flow + jitter"],
  ];
  evidenceGrid.innerHTML = evidence
    .map(
      ([label, value, detail]) => `
        <div class="evidence-card">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
          <span>${escapeHtml(detail)}</span>
        </div>
      `,
    )
    .join("");
}

function renderDownloads(downloads) {
  const links = [
    ["summary_csv", "summary.csv"],
    ["frame_csv", "frame_metrics.csv"],
    ["result_json", "result.json"],
  ];
  downloadRow.innerHTML = links
    .filter(([key]) => downloads?.[key])
    .map(
      ([key, label]) =>
        `<a class="download-link" href="${escapeHtml(downloads[key])}" download>${label} ↗</a>`,
    )
    .join("");
}

function renderResult(payload) {
  const result = payload.result;
  const score = Number(result.weighted_score_0_100);
  const coverage = result.coverage ?? "0/5";
  const mode = modeLabels[result.evaluation_mode] ?? result.evaluation_mode ?? "REPORT";
  emptyReport.classList.add("is-hidden");
  reportContent.classList.remove("is-hidden");
  reportMode.textContent = mode;
  reportMode.classList.add("success");
  overallScore.textContent = Number.isFinite(score) ? score.toFixed(1) : "—";
  coverageValue.textContent = coverage;
  scoreCaption.textContent = `${result.weighted_score_weight_coverage ?? 0}% weight covered · ${result.status}`;
  const covered = Number.parseInt(coverage, 10) || 0;
  coverageRing.style.setProperty("--coverage", `${(covered / 5) * 100}%`);
  renderRadar(result);
  renderCategories(result);
  renderEvidence(result);
  renderDownloads(payload.downloads);
  document.querySelector("#report-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setBusy(isBusy) {
  evaluateButton.disabled = isBusy;
  evaluateButton.querySelector("span:first-child").textContent = isBusy ? "评估中..." : "开始评估";
}

let progressTimer = null;
let progressStageTimer = null;
let progressStartedAt = 0;

function setProgressStage(stage, label) {
  progressLabel.textContent = label;
  progressSteps.forEach((step) => {
    step.classList.toggle("is-active", step.dataset.progressStep === stage);
  });
}

function startProgress() {
  processProgress.classList.remove("is-hidden");
  progressStartedAt = Date.now();
  progressBar.classList.remove("is-complete");
  setProgressStage("upload", "正在读取上传文件");
  progressTime.textContent = "00:00";
  progressTimer = window.setInterval(() => {
    const elapsed = Math.floor((Date.now() - progressStartedAt) / 1000);
    progressTime.textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(
      elapsed % 60,
    ).padStart(2, "0")}`;
  }, 250);

  const stages = [
    ["sample", "正在抽取关键帧"],
    ["models", "正在运行本地评估模型"],
    ["report", "正在整理评分报告"],
  ];
  let stageIndex = 0;
  progressStageTimer = window.setInterval(() => {
    const [stage, label] = stages[Math.min(stageIndex, stages.length - 1)];
    setProgressStage(stage, label);
    stageIndex += 1;
    if (stageIndex >= stages.length) {
      window.clearInterval(progressStageTimer);
      progressStageTimer = null;
    }
  }, 1800);
}

function stopProgress(completed = false) {
  if (progressTimer) window.clearInterval(progressTimer);
  if (progressStageTimer) window.clearInterval(progressStageTimer);
  progressTimer = null;
  progressStageTimer = null;
  if (completed) {
    setProgressStage("report", "评估完成");
    progressBar.classList.add("is-complete");
    progressTime.textContent = "DONE";
  } else {
    processProgress.classList.add("is-hidden");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (window.queueMode) return;
  setBusy(true);
  startProgress();
  setFormNote("正在读取视频、抽帧并运行本地模型，请稍候...");
  try {
    const formData = new FormData(form);
    formData.set(
      "calculate_lpips",
      form.querySelector('[name="calculate_lpips"]').checked ? "true" : "false",
    );
    const response = await fetch("/api/evaluate", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(describeApiError(payload.detail, response.status));
    }
    if (!payload.result) {
      throw new Error("服务返回了不完整的评估结果。");
    }
    renderResult(payload);
    stopProgress(true);
    setFormNote(`评估完成：${payload.run_id}`, "success");
  } catch (error) {
    stopProgress(false);
    setFormNote(error.message || "评估失败，请检查输入文件。", "error");
  } finally {
    setBusy(false);
  }
});

loadModels();

window.queueMode = true;

const queueStageLabels = {
  queued: ["upload", "排队等待"],
  preparing: ["upload", "准备输入文件"],
  sampling: ["sample", "抽取关键帧"],
  models: ["models", "运行本地模型"],
  report: ["report", "整理评估报告"],
  completed: ["report", "评估完成"],
  failed: ["report", "评估失败"],
  canceled: ["upload", "任务已取消"],
};
const queueStatusLabels = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  canceled: "已取消",
};
const queueTerminalStatuses = new Set(["completed", "failed", "canceled"]);
const queueKnownStatuses = new Map();
let selectedJobId = null;
let queueRefreshInFlight = false;

function setQueueBusy(isBusy) {
  evaluateButton.disabled = isBusy;
  evaluateButton.querySelector("span:first-child").textContent = isBusy
    ? "上传中..."
    : "加入队列";
}

function updateQueueProgressPanel(job) {
  if (!job) {
    processProgress.classList.add("is-hidden");
    return;
  }
  const [stage, label] =
    queueStageLabels[job.stage] ?? queueStageLabels.queued;
  progressLabel.textContent = label;
  progressSteps.forEach((step) => {
    step.classList.toggle("is-active", step.dataset.progressStep === stage);
  });
  processProgress.classList.remove("is-hidden");
  progressBar.classList.remove("is-complete");
  progressBar.style.animation = "none";
  progressBar.style.width = `${Math.max(
    0,
    Math.min(100, Number(job.progress ?? 0) * 100),
  )}%`;
  if (job.status === "completed") {
    progressBar.classList.add("is-complete");
  }
  if (queueTerminalStatuses.has(job.status)) {
    progressTime.textContent = queueStatusLabels[job.status] ?? job.status;
    return;
  }
  const startedAt = Date.parse(job.started_at ?? "");
  if (!Number.isFinite(startedAt)) {
    progressTime.textContent = "00:00";
    return;
  }
  const elapsed = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
  progressTime.textContent = `${String(Math.floor(elapsed / 60)).padStart(
    2,
    "0",
  )}:${String(elapsed % 60).padStart(2, "0")}`;
}

function queueStatusText(status) {
  return queueStatusLabels[status] ?? "未知状态";
}

function renderQueue(payload) {
  const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
  const active = jobs.find((job) => job.status === "running");
  queueSummary.textContent = `${active ? 1 : 0} 个运行 / ${
    payload.queued_count ?? 0
  } 个等待`;
  queueActive.classList.toggle("is-hidden", !active);
  if (active) {
    activeJobName.textContent = active.name || active.job_id;
    activeJobStage.textContent =
      queueStageLabels[active.stage]?.[1] ?? active.stage;
    activeJobStatus.textContent = queueStatusText(active.status);
    activeJobStatus.className = `queue-status ${escapeHtml(active.status)}`;
    activeJobProgress.style.width = `${Math.max(
      0,
      Math.min(100, Number(active.progress ?? 0) * 100),
    )}%`;
  }

  const visibleJobs = jobs
    .filter((job) => job.status !== "running")
    .slice(0, 10);
  queueList.innerHTML = visibleJobs
    .map((job, index) => {
      const position = job.queue_position ?? index + 1;
      const actions = [];
      if (job.status === "queued") {
        actions.push(["cancel", "取消"]);
      }
      if (job.status === "failed" || job.status === "canceled") {
        actions.push(["retry", "重试"]);
      }
      if (job.status !== "running") {
        actions.push(["delete", "删除"]);
      }
      const actionMarkup = actions
        .map(
          ([action, label]) =>
            `<button class="queue-item-action ${action === "delete" ? "is-danger" : ""}" type="button" data-action="${action}" data-job-id="${escapeHtml(job.job_id)}">${label}</button>`,
        )
        .join("");
      return `
        <div class="queue-item-row">
          <span class="queue-item-index">${String(position).padStart(2, "0")}</span>
          <button class="queue-item" type="button" data-job-id="${escapeHtml(job.job_id)}">
            <span class="queue-item-copy">
              <strong>${escapeHtml(job.name || job.job_id)}</strong>
              <small>${escapeHtml(queueStageLabels[job.stage]?.[1] ?? queueStatusText(job.status))}</small>
            </span>
            <span class="queue-item-status ${escapeHtml(job.status)}">${queueStatusText(job.status)}</span>
          </button>
          <div class="queue-item-actions">${actionMarkup}</div>
        </div>
      `;
    })
    .join("");
  queueEmpty.classList.toggle("is-hidden", jobs.length > 0);

  queueList.querySelectorAll(".queue-item").forEach((item) => {
    item.addEventListener("click", () => selectQueueJob(item.dataset.jobId));
  });
  queueList.querySelectorAll(".queue-item-action").forEach((item) => {
    item.addEventListener("click", (event) => {
      event.stopPropagation();
      mutateQueueJob(item.dataset.jobId, item.dataset.action);
    });
  });
}

async function getQueueJob(jobId, shouldScroll = false) {
  if (!jobId) return null;
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(describeApiError(payload.detail, response.status));
  }
  selectedJobId = payload.job_id;
  updateQueueProgressPanel(payload);
  if (payload.result) {
    renderResult(payload);
    if (shouldScroll) {
      document.querySelector("#report-panel").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }
  return payload;
}

async function selectQueueJob(jobId) {
  try {
    const payload = await getQueueJob(jobId, true);
    if (payload?.status === "completed") {
      setFormNote(`已加载报告 / ${payload.name}`, "success");
    } else if (payload?.error) {
      setFormNote(payload.error, "error");
    } else {
      setFormNote(`${queueStatusText(payload?.status)} / ${payload?.name}`);
    }
  } catch (error) {
    setFormNote(error.message || "无法加载队列任务。", "error");
  }
}

async function mutateQueueJob(jobId, action) {
  if (!jobId) return;
  if (
    action === "delete" &&
    !window.confirm("确定删除这个任务及其已保存的文件吗？")
  ) {
    return;
  }
  try {
    const response =
      action === "delete"
        ? await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
            method: "DELETE",
          })
        : await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action }),
          });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(describeApiError(payload.detail, response.status));
    }
    if (selectedJobId === jobId && action === "delete") {
      selectedJobId = null;
      processProgress.classList.add("is-hidden");
    }
    await refreshQueue();
  } catch (error) {
    setFormNote(error.message || "无法更新队列任务。", "error");
  }
}

async function refreshQueue() {
  if (queueRefreshInFlight) return;
  queueRefreshInFlight = true;
  try {
    const response = await fetch("/api/jobs?limit=20");
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(describeApiError(payload.detail, response.status));
    }
    renderQueue(payload);
    const selected = payload.jobs?.find(
      (job) => job.job_id === selectedJobId,
    );
    if (selected) {
      updateQueueProgressPanel(selected);
      const previousStatus = queueKnownStatuses.get(selected.job_id);
      if (
        selected.status === "completed" &&
        previousStatus !== "completed"
      ) {
        await getQueueJob(selected.job_id);
        setFormNote(`评估完成 / ${selected.name}`, "success");
      } else if (
        selected.status === "failed" &&
        previousStatus !== "failed"
      ) {
        setFormNote(selected.error || "Evaluation failed.", "error");
      }
      queueKnownStatuses.set(selected.job_id, selected.status);
    }
  } catch (error) {
    setFormNote(error.message || "无法刷新处理队列。", "error");
  } finally {
    queueRefreshInFlight = false;
  }
}

refreshQueueButton.addEventListener("click", refreshQueue);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setQueueBusy(true);
  setFormNote("正在上传文件并加入处理队列...");
  try {
    const formData = new FormData(form);
    formData.set(
      "calculate_lpips",
      form.querySelector('[name="calculate_lpips"]').checked ? "true" : "false",
    );
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(describeApiError(payload.detail, response.status));
    }
    if (!payload.job_id) {
      throw new Error("队列没有返回任务编号。");
    }
    selectedJobId = payload.job_id;
    queueKnownStatuses.set(payload.job_id, payload.status);
    updateQueueProgressPanel(payload);
    setFormNote(
      `已加入队列 / 第 ${payload.queue_position ?? "下一个"} 位 / ${payload.name}`,
      "success",
    );
    await refreshQueue();
  } catch (error) {
    setFormNote(error.message || "无法创建队列任务。", "error");
  } finally {
    setQueueBusy(false);
  }
});

setQueueBusy(false);
refreshQueue();
window.setInterval(refreshQueue, 1200);
