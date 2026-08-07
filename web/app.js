const form = document.querySelector("#evaluation-form");
const evaluateButton = document.querySelector("#evaluate-button");
const newEvaluationButton = document.querySelector("#new-evaluation");
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
const qwenFeedback = document.querySelector("#qwen-feedback");
const wangxingResult = document.querySelector("#wangxing-result");
const wangxingReadiness = document.querySelector("#wangxing-au-readiness");
const processProgress = document.querySelector("#process-progress");
const progressLabel = document.querySelector("#progress-label");
const progressTime = document.querySelector("#progress-time");
const progressBar = document.querySelector("#progress-bar");
const progressSteps = [...document.querySelectorAll("[data-progress-step]")];
const queueSummary = document.querySelector("#queue-summary");
const queueActive = document.querySelector("#queue-active");
const activeJobName = document.querySelector("#active-job-name");
const activeJobStage = document.querySelector("#active-job-stage");
const activeJobTime = document.querySelector("#active-job-time");
const activeJobStatus = document.querySelector("#active-job-status");
const activeJobCancel = document.querySelector("#active-job-cancel");
const activeJobProgress = document.querySelector("#active-job-progress");
const queueList = document.querySelector("#queue-list");
const queueEmpty = document.querySelector("#queue-empty");
const refreshQueueButton = document.querySelector("#refresh-queue");
const queueSearch = document.querySelector("#queue-search");
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
  if (value === "inf" || value === "+inf" || value === "Infinity") return "∞";
  if (value === "-inf" || value === "-Infinity") return "-∞";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toFixed(digits);
}

function normalizeScore(value) {
  if (value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : null;
}

function categoryScore(key, category) {
  const metrics = category?.metrics ?? {};
  const valueByCategory = {
    identity: metrics.score_0_1,
    texture: metrics.score_0_1,
    expression: category?.score_0_1 ?? metrics.score_0_1,
    temporal: metrics.stability_score_0_1 ?? metrics.score_0_1,
    aesthetics: metrics.manual_score_0_to_1 ?? category?.score_0_1,
  };
  return normalizeScore(category?.score_0_1 ?? valueByCategory[key]);
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
    categoryScore(key, result.categories?.[key]),
  );
  const labels = ["身份一致性", "质感细节", "表情准确", "时间稳定", "美学质量"];
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
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="五维评分雷达图">
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
      ? id === "reference-video"
        ? `已选择 ${files.length} 段参考视频`
        : `${files.length} reference image${files.length === 1 ? "" : "s"}`
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
          if (!input.multiple) {
            name.textContent = `${file.name}${seconds}`;
          }
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
        model.name.includes("ETVA VLM") && model.ready && model.service_active === false
          ? "CACHED"
          : model.status === "ready"
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

function renderWangxingReadiness(payload) {
  if (!wangxingReadiness) return;
  const ready = Boolean(payload?.ready);
  wangxingReadiness.textContent = ready ? "READY" : "TRAIN FIRST";
  wangxingReadiness.className = `au-readiness ${ready ? "ready" : "offline"}`;
  wangxingReadiness.title = payload?.note ?? "";
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("model endpoint unavailable");
    const payload = await response.json();
    renderModels(payload);
    renderWangxingReadiness(payload.wangxing_au);
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
      const score = categoryScore(key, category);
      const scoreText = score === null ? "—" : (score * 100).toFixed(2);
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

function renderQwenFeedback(result) {
  if (!qwenFeedback) return;
  const judge = result.etva_judge ?? {};
  const feedback = judge.feedback ?? {};
  const problems = Array.isArray(feedback.problems)
    ? feedback.problems.filter(Boolean)
    : [];
  const suggestions = Array.isArray(feedback.suggestions)
    ? feedback.suggestions.filter(Boolean)
    : [];
  const isAvailable = judge.status === "available";
  const unavailableReason = String(judge.reason ?? "");
  const isServiceActive =
    judge.service_active === true ||
    judge.failure_kind === "invalid_response" ||
    /service is connected/i.test(unavailableReason);
  const isCachedButOffline =
    !isAvailable &&
    (judge.service_active === false ||
      judge.failure_kind === "service_unavailable" ||
      /weights are cached, but .*HTTP service is not connected/i.test(unavailableReason));
  const hasInvalidResponse = !isAvailable && isServiceActive && !isCachedButOffline;
  const judgeModelLabel = String(judge.model || "qwen2-vl-2b-awq")
    .replaceAll("-awq", " AWQ")
    .replaceAll("qwen2-vl-2b", "Qwen2-VL-2B")
    .replaceAll("qwen2.5-vl-3b", "Qwen2.5-VL-3B");
  const unavailableMessage = isCachedButOffline
    ? "Qwen 权重已下载，但 Judge 服务未连接；下载模型不会自动启动 HTTP 服务。"
    : hasInvalidResponse
      ? "Qwen Judge 服务已连接，但未返回可解析的有效诊断，请检查服务日志。"
      : "Qwen 模型未返回文字诊断，请重新评估或检查 VLM 服务。";
  const windowCount = Number(judge.metrics?.window_count);
  const statusText = isAvailable
    ? Number.isFinite(windowCount)
      ? `${judgeModelLabel} · 已分析 ${windowCount} 个时间窗口`
      : `${judgeModelLabel} · 已完成复核`
    : isCachedButOffline
      ? `${judgeModelLabel} · 已下载，当前未连接`
      : hasInvalidResponse
        ? `${judgeModelLabel} · Judge 已连接，但未返回有效诊断`
      : `${judgeModelLabel} · 当前未返回诊断`;

  qwenFeedback.classList.remove("is-hidden");
  qwenFeedback.innerHTML = `
    <div class="qwen-feedback-head">
      <div>
        <span class="qwen-feedback-kicker">QWEN2-VL-2B AWQ / VIDEO REVIEW</span>
        <h3>视频问题与调整建议</h3>
        <p>${escapeHtml(statusText)}</p>
      </div>
      <span class="qwen-feedback-badge ${isAvailable ? "ready" : "muted"}">
        ${
          isAvailable
            ? "AI REVIEW"
              : isCachedButOffline
                ? "已下载 / 未连接"
                : hasInvalidResponse
                  ? "Judge 已连接 / 诊断失败"
                : "诊断不可用"
        }
      </span>
    </div>
    <div class="qwen-feedback-grid">
      <div class="qwen-feedback-column problem">
        <span class="qwen-feedback-label">模型发现的问题</span>
        ${
          problems.length
            ? `<ul>${problems
                .slice(0, 6)
                .map((item) => `<li>${escapeHtml(item)}</li>`)
                .join("")}</ul>`
            : `<p class="qwen-feedback-empty">${
                isAvailable
                  ? "未发现明确问题。"
                  : unavailableMessage
              }</p>`
        }
      </div>
      <div class="qwen-feedback-column suggestion">
        <span class="qwen-feedback-label">可以尝试的调整</span>
        ${
          suggestions.length
            ? `<ul>${suggestions
                .slice(0, 6)
                .map((item) => `<li>${escapeHtml(item)}</li>`)
                .join("")}</ul>`
            : `<p class="qwen-feedback-empty">${
                isAvailable
                  ? "暂无额外调整建议。"
                  : hasInvalidResponse
                    ? "请检查 Judge 服务日志，确认模型返回的是有效 JSON 后再重新评估。"
                    : "启动 Qwen Judge 服务后重新评估，这里会显示具体调整方向。"
              }</p>`
        }
      </div>
    </div>
  `;
}

function renderEvidence(result) {
  const categories = result.categories ?? {};
  const textureCategory = categories.texture ?? {};
  const texture = textureCategory.metrics ?? {};
  const identity = categories.identity?.metrics ?? {};
  const expression = categories.expression?.metrics ?? {};
  const temporal = categories.temporal?.metrics ?? {};
  const aesthetics = categories.aesthetics?.metrics ?? {};
  const identitySource = String(categories.identity?.reference_source ?? "");
  const identityReferenceLabel = identitySource.includes("gt_video")
    ? "ArcFace / 参考图 + 参考视频 + GT"
    : identitySource.includes("reference_video")
      ? "ArcFace / 参考图 + 参考视频"
      : identitySource.includes("reference_image")
        ? "ArcFace / 参考图"
        : "ArcFace / 参考素材";
  const fullReference = textureCategory.mode === "full_reference";
  const groundTruthStatus = String(
    textureCategory.ground_truth_status ??
      (result.ground_truth_video ? "uploaded_but_unusable" : "not_uploaded"),
  );
  const groundTruthUploaded =
    groundTruthStatus === "used" ||
    groundTruthStatus === "uploaded_but_unusable" ||
    Boolean(result.ground_truth_video);
  const textureReferenceDetail = fullReference
    ? "GT 参考"
    : groundTruthUploaded
      ? "GT 已上传 / 未通过对齐"
      : "无 GT";
  const manualAesthetic = aesthetics.manual_score_0_to_1;
  const vbenchAesthetic = aesthetics.vbench_aesthetic_quality_0_to_1;
  const aestheticValue = manualAesthetic;
  const aestheticLabel = manualAesthetic === null || manualAesthetic === undefined
    ? "AESTHETIC / MANUAL REQUIRED"
    : "AESTHETIC / MANUAL";
  const aestheticChineseLabel = manualAesthetic === null || manualAesthetic === undefined
    ? "美学评分 / 需人工"
    : "人工审美评分";
  const aestheticDetail = manualAesthetic === null || manualAesthetic === undefined
    ? vbenchAesthetic === null || vbenchAesthetic === undefined
      ? "暂无正式分数"
      : `VBench 辅助 ${formatNumber(vbenchAesthetic)}`
    : "人工优先";
  const evidence = [
    ["IDENTITY / MEAN", "身份一致性 / 均值", formatNumber(identity.mean_similarity), identityReferenceLabel, "higher"],
    [fullReference ? "PSNR / dB" : "TEXTURE / SCORE", fullReference ? "峰值信噪比" : "纹理质量 / 分数", fullReference ? formatNumber(texture.psnr_db, 2) : formatNumber(texture.score_0_1), textureReferenceDetail, "higher"],
    [fullReference ? "SSIM" : "MANIQA", fullReference ? "结构相似性" : "图像质量评分", fullReference ? formatNumber(texture.ssim, 4) : formatNumber(texture.maniqa), fullReference ? "GT 参考" : groundTruthUploaded ? "GT 已上传 / 未通过对齐" : "可选图像质量指标", "higher"],
    [fullReference ? "LPIPS" : "MUSIQ", fullReference ? "感知距离" : "无参考质量评分", fullReference ? formatNumber(texture.lpips, 4) : formatNumber(texture.musiq), fullReference ? "GT 参考" : groundTruthUploaded ? "GT 已上传 / 未通过对齐" : "可选图像质量指标", fullReference ? "lower" : "higher"],
    ["TEXT / VIDEO", "文本 / 视频一致性", formatNumber(expression.text_video_alignment), "CLIP 基线", "higher"],
    ["TEMPORAL / STABILITY", "时间稳定性", formatNumber(temporal.stability_score_0_1), "光流 + 抖动", "higher"],
  ];
  evidence.push([
    aestheticLabel,
    aestheticChineseLabel,
    formatNumber(aestheticValue),
    aestheticDetail,
    "higher",
  ]);
  evidenceGrid.innerHTML = evidence
    .map(
      ([label, chineseLabel, value, detail, direction]) => {
        const isHigherBetter = direction === "higher";
        const directionArrow = isHigherBetter ? "↑" : "↓";
        const directionText = isHigherBetter ? "越高越好" : "越低越好";
        return `
        <div class="evidence-card">
          <div class="evidence-card-label">
            <span>${escapeHtml(label)}</span>
            <small>${escapeHtml(chineseLabel)}</small>
          </div>
          <div class="evidence-value">
            <strong>${escapeHtml(value)}</strong>
            <span class="metric-direction ${isHigherBetter ? "is-higher" : "is-lower"}" title="${directionText}" aria-label="${directionText}">${directionArrow}</span>
          </div>
          <span class="evidence-direction">${directionText}</span>
          <span>${escapeHtml(detail)}</span>
        </div>
        `;
      },
    )
    .join("") +
    (groundTruthUploaded && !fullReference
      ? `<div class="evidence-note"><strong>GT 已上传，但未用于 PSNR / SSIM / LPIPS。</strong><span>GT 与结果视频已按共同时间区间进行比较。</span></div>`
      : "");
}

function clampUnit(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return Math.max(0, Math.min(1, number));
}

const expressionLabels = {
  auto: "自动判断",
  unknown: "数据不足，无法可靠归类",
  smile: "微笑",
  anger: "愤怒",
  annoyance: "烦躁",
  surprise: "惊讶",
  fear: "恐惧",
  sadness: "悲伤",
};

expressionLabels.disgust = "厌恶";

function expressionLabel(value) {
  const key = String(value ?? "").toLowerCase();
  return expressionLabels[key] ?? (key || "未指定");
}

function describeAuActivity(auIds) {
  const ids = new Set(auIds.map(Number));
  const hasSmile = ids.has(6) || ids.has(12);
  const hasOpenMouth = ids.has(25) || ids.has(26);
  const hasTightEyes = ids.has(4) || ids.has(7);
  const hasTightMouth = ids.has(15) || ids.has(17);
  const hasRaisedBrows = ids.has(1) || ids.has(2) || ids.has(5);
  const hasNoseWrinkle = ids.has(9);
  const hasWideLips = ids.has(20);

  if (hasTightEyes && hasTightMouth) {
    return {
      label: "眉眼收紧，嘴唇抿住",
      detail: "面部更接近不悦或烦躁状态",
    };
  }
  if (hasTightEyes) {
    return {
      label: "眉眼收紧",
      detail: "眉毛下压，眼周显得更紧",
    };
  }
  if (hasSmile && hasOpenMouth) {
    return {
      label: "微笑并张嘴",
      detail: "嘴角上扬，脸颊提起，同时嘴巴张开",
    };
  }
  if (hasSmile) {
    return {
      label: "微笑",
      detail: "嘴角上扬，脸颊提起",
    };
  }
  if (hasOpenMouth) {
    return {
      label: "张嘴",
      detail: "嘴唇分开，下颌下落",
    };
  }
  if (hasTightMouth) {
    return {
      label: "抿嘴或嘴角下压",
      detail: "嘴部出现收紧或下压动作",
    };
  }
  if (hasRaisedBrows) {
    return {
      label: "抬眉或睁大眼",
      detail: "眉眼区域出现上提动作",
    };
  }
  if (hasNoseWrinkle) {
    return {
      label: "鼻翼收紧",
      detail: "鼻部出现轻微收紧动作",
    };
  }
  if (hasWideLips) {
    return {
      label: "嘴唇拉宽",
      detail: "嘴部横向拉伸",
    };
  }
  return {
    label: "无明显表情变化",
    detail: "没有足够强度和持续时间的表情动作通过显著性阈值",
  };
}

function collectAuTemporalSegments(temporalEvents) {
  const frameCount = Math.max(1, Number(temporalEvents?.frame_count) || 1);
  const meshMouthEvents =
    temporalEvents?.face_mesh?.status === "available"
      ? temporalEvents.face_mesh.mouth_open?.events ?? []
      : null;
  const intervals = [];
  Object.entries(temporalEvents?.per_au ?? {}).forEach(([auId, summary]) => {
    (Array.isArray(summary?.events) ? summary.events : []).forEach((event) => {
      const isSalient =
        event.salient === true ||
        (event.salient === undefined &&
          Number(event.peak_intensity) >= 0.5 &&
          Number(event.duration_ratio) >= 0.05);
      if (!isSalient) return;
      if (meshMouthEvents && (Number(auId) === 25 || Number(auId) === 26)) {
        const overlapsMouthMesh = meshMouthEvents.some(
          (meshEvent) =>
            Number(meshEvent.start_frame) <= Number(event.end_frame) &&
            Number(meshEvent.end_frame) >= Number(event.start_frame) &&
            meshEvent.salient !== false,
        );
        if (!overlapsMouthMesh) return;
      }
      const start = Number(event.start_frame);
      const end = Number(event.end_frame);
      if (
        Number.isFinite(start) &&
        Number.isFinite(end) &&
        end >= start
      ) {
        intervals.push({ start, end, auId: Number(auId) });
      }
    });
  });
  if (!intervals.length) return [];

  const mergeGap = Math.max(2, Math.round(frameCount * 0.04));
  intervals.sort((left, right) => left.start - right.start || left.end - right.end);
  const groups = [];
  intervals.forEach((interval) => {
    const current = groups[groups.length - 1];
    if (!current || interval.start > current.end + mergeGap) {
      groups.push({
        start: interval.start,
        end: interval.end,
        auIds: new Set([interval.auId]),
      });
      return;
    }
    current.end = Math.max(current.end, interval.end);
    current.auIds.add(interval.auId);
  });

  return groups.slice(0, 5).map((group) => ({
    ...group,
    activity: describeAuActivity([...group.auIds]),
    startPosition: clampUnit(group.start / Math.max(frameCount - 1, 1)),
    endPosition: clampUnit(group.end / Math.max(frameCount - 1, 1)),
  }));
}

function formatTemporalRange(segment) {
  const start = Number(segment.start);
  const end = Number(segment.end);
  const startPercent = Math.round((segment.startPosition ?? 0) * 100);
  const endPercent = Math.round((segment.endPosition ?? 0) * 100);
  const frameText =
    start === end ? `第 ${start} 帧` : `第 ${start}–${end} 帧`;
  return `${frameText} · 视频进度 ${startPercent}%–${endPercent}%`;
}

function renderAuTemporalEvidence(au) {
  const temporalEvents = au.temporal_events ?? {};
  const segments = collectAuTemporalSegments(temporalEvents);
  const meshEnabled = temporalEvents.face_mesh?.status === "available";
  const expectedClass = au.expected_expression_class;
  const selectedClass = au.selected_expression_class ?? "auto";
  const expressionContext = expectedClass
    ? `目标表情：${expressionLabel(expectedClass)}`
    : `模型自动归类：${expressionLabel(selectedClass)}`;
  const overallIds = [
    ...new Set(segments.flatMap((segment) => [...segment.auIds])),
  ];
  const overallActivity = describeAuActivity(overallIds);
  const evidenceText = segments.length
    ? `检测到 ${segments.length} 个主要时段`
    : "未检测到明显表情变化";
  return `
    <div class="wangxing-result-temporal">
      <div class="wangxing-result-section-head">
        <span>AU EVENT STATISTICS / 当前视频表情事件</span>
        <small>${escapeHtml(
          `${meshEnabled ? "Face Mesh + AU 交叉验证 · " : "AU 单独分析 · "}只描述当前视频面部变化`,
        )}</small>
      </div>
      <div class="au-expression-summary">
        <div>
          <span class="au-expression-summary-label">视频中最明显的表情</span>
          <strong>${escapeHtml(overallActivity.label)}</strong>
          <p>${escapeHtml(overallActivity.detail)} · ${escapeHtml(expressionContext)}</p>
        </div>
        <span class="au-expression-summary-score">${escapeHtml(evidenceText)}</span>
      </div>
      ${
        segments.length
          ? `
            <div class="au-expression-timeline">
              ${segments
                .map(
                  (segment, index) => `
                    <article class="au-expression-segment">
                      <span class="au-expression-segment-index">0${index + 1}</span>
                      <div>
                        <strong>${escapeHtml(segment.activity.label)}</strong>
                        <p>${escapeHtml(formatTemporalRange(segment))}</p>
                        <small>${escapeHtml(segment.activity.detail)}</small>
                      </div>
                    </article>
                  `,
                )
                .join("")}
            </div>
          `
          : '<div class="au-expression-empty">未检测到明显的表情变化。</div>'
      }
      <p class="au-presence-note">
        以上内容只描述当前视频自身的面部 AU 表情时段；专项动态一致性来自训练 CSV 的统计分布，不比较参考视频或某条训练视频的时间轴。
      </p>
    </div>
  `;
}

function renderWangxingResult(result) {
  if (!wangxingResult) return;
  const payload = result.wangxing_au;
  if (!payload) {
    wangxingResult.classList.add("is-hidden");
    return;
  }
  if (payload.schema_version === "wangxing_specialization_v1") {
    renderWangxingSpecializationResult(payload);
    return;
  }
  wangxingResult.classList.remove("is-hidden");
  wangxingResult.innerHTML = `
    <div class="wangxing-result-head">
      <div>
        <span class="wangxing-result-kicker">TARGET SPECIALIZATION / WANG XING</span>
        <h3>王兴专项需要重新运行</h3>
      </div>
      <span class="wangxing-result-status review">LEGACY RESULT</span>
    </div>
    <p class="wangxing-result-note">
      旧版专项结果不再展示动作、训练时序或单条训练视频对齐结论。
      请使用当前的身份门控与表情画像报告。
    </p>
  `;
  if (payload.status !== "available") {
    wangxingResult.classList.remove("is-hidden");
    const notApplicable = payload.status === "not_applicable";
    wangxingResult.innerHTML = `
      <div class="wangxing-result-head">
        <div>
          <span class="wangxing-result-kicker">TARGET SPECIALIZATION / WANG XING AU</span>
          <h3>${notApplicable ? "王兴专项未启用" : "王兴特化评估暂不可用"}</h3>
        </div>
        <span class="wangxing-result-status review">${
          notApplicable ? "NOT APPLICABLE / 不适用" : "UNAVAILABLE"
        }</span>
      </div>
      <p class="wangxing-result-note">${escapeHtml(payload.reason ?? "AU 评估未运行。")}</p>
    `;
    return;
  }
  const targeted = payload.wangxing_targeted ?? {};
  const au = payload.au_compliance ?? {};
  const fusion = payload.fusion ?? {};
  const quality = au.quality?.generated ?? au.generated_au?.quality ?? {};
  const identity = normalizeScore(
    payload.identity_preservation?.metrics?.score_0_1,
  );
  const score = normalizeScore(
    targeted.wangxing_expression_fit_score_0_1,
  );
  const rawDecision =
    fusion.decision === "allow" && targeted.decision === "allow"
      ? "allow"
      : "review";
  const uncertain = targeted.evidence_quality_status === "uncertain";
  const decision =
    rawDecision === "block" ||
    (uncertain && rawDecision === "allow")
      ? "review"
      : rawDecision;
  const decisionLabel = {
    allow: "ALLOW / 符合",
    review: "REVIEW / 需复核",
  }[decision] ?? "REVIEW / 复核";
  const scoreText = score === null ? "—" : `${(score * 100).toFixed(1)}`;
  const selectedClass = au.selected_expression_class ?? "auto";
  const expectedClass =
    au.expected_expression_class ??
    targeted.expected_expression_class;
  const selectedClassScore = au.class_scores?.[selectedClass] ?? {};
  const exactProfileMatch =
    selectedClassScore.exact_sequence_match === true;
  const exactProfileMatchSource =
    selectedClassScore.exact_sequence_match_source ??
    au.exact_profile_match_source;
  const classContext = expectedClass
    ? `目标：${expressionLabel(expectedClass)}`
    : `自动归类：${expressionLabel(selectedClass)}`;
  const personal = normalizeScore(targeted.evidence?.personal_au);
  const facialDynamics = normalizeScore(
    targeted.evidence?.facial_dynamics ??
      au.facial_expression_dynamics_score_0_1,
  );
  const eventAggregate = au.temporal_events?.aggregate ?? {};
  const eventCount = Number(eventAggregate.event_count ?? 0);
  const eventActiveRatio = normalizeScore(eventAggregate.active_ratio);
  const eventStatistics = Number.isFinite(eventCount)
    ? `${Math.max(0, Math.round(eventCount))} 次`
    : "—";
  const leakage = normalizeScore(targeted.evidence?.leakage_risk);
  const confidence = normalizeScore(
    targeted.evidence_confidence_0_1 ??
      au.evidence_confidence_0_1,
  );
  const autoClassificationReason = String(
    au.auto_classification_reason ?? "",
  );
  const autoClassificationNote =
    /at least two|too few|not found|not ready/i.test(autoClassificationReason)
      ? "原版 AU 情绪数据未达到自动分类条件，至少需要两类情绪且每类至少 3 个样本。"
      : autoClassificationReason ||
        "原版 AU 情绪数据不足，暂不输出确定情绪。";
  const thresholds = targeted.thresholds ?? {};
  const validFrameRatio = normalizeScore(quality.valid_frame_ratio);
  const qualityStatus = String(
    targeted.evidence_quality_status ??
      au.evidence_quality_status ??
      quality.status ??
      "available",
  );
  const qualityLabel = {
    pass: "FACE QUALITY / GOOD",
    partial: "FACE QUALITY / PARTIAL",
    uncertain: "FACE QUALITY / UNCERTAIN",
    not_available: "FACE QUALITY / NOT AVAILABLE",
    available: "FACE QUALITY / AVAILABLE",
  }[qualityStatus] ?? "FACE QUALITY / CHECK";
  const reasonLabels = {
    missing_personal_au: "缺少个人 AU",
    missing_facial_dynamics: "缺少面部表情动态统计",
    automatic_expression_class_unavailable: "自动表情分类不可用",
    face_quality_low: "人脸质量不足",
    evidence_quality_low: "证据质量不足",
    wangxing_au_below_threshold: "AU 画像偏离",
    facial_dynamics_below_threshold: "面部表情动态统计偏离",
    identity_leakage_risk: "身份偏离风险",
  };
  const reasons = (targeted.decision_reasons ?? [])
    .map((reason) => reasonLabels[reason] ?? reason)
    .join(" / ");
  const missingEvidence = [
    ...(fusion.missing_evidence ?? []),
    ...(targeted.missing_evidence ?? []),
  ]
    .filter((item, index, values) => values.indexOf(item) === index)
    .map((item) => reasonLabels[item] ?? item)
    .join(" / ");
  const evidenceCoverage = normalizeScore(
    targeted.score_weight_coverage ??
      targeted.evidence_coverage_0_1,
  );
  const formatEvidence = (value) =>
    value === null ? "—" : `${(value * 100).toFixed(1)}`;
  const thresholdNotes = [];
  if (
    personal !== null &&
    Number.isFinite(Number(thresholds.personal_au)) &&
    personal < Number(thresholds.personal_au)
  ) {
    thresholdNotes.push(
      `王兴 AU ${formatEvidence(personal)} < 阈值 ${formatEvidence(
        thresholds.personal_au,
      )}`,
    );
  }
  if (
    leakage !== null &&
    Number.isFinite(Number(thresholds.leakage)) &&
    leakage >= Number(thresholds.leakage)
  ) {
    thresholdNotes.push(
      `身份泄漏风险 ${formatEvidence(leakage)} >= 阈值 ${formatEvidence(
        thresholds.leakage,
      )}`,
    );
  }
  const decisionNote = [
    selectedClass === "unknown"
      ? autoClassificationNote
      : "",
    exactProfileMatch
      ? exactProfileMatchSource === "video_hash"
        ? "生成视频与王兴画像中的训练视频精确匹配。"
        : "生成 AU 序列与王兴画像中的训练序列精确匹配。"
      : "",
    reasons,
    missingEvidence ? `缺少证据：${missingEvidence}` : "",
    evidenceCoverage !== null
      ? `证据覆盖 ${formatEvidence(evidenceCoverage)}`
      : "",
    thresholdNotes.join(" / "),
  ]
    .filter(Boolean)
    .join(" ");
  const identityText =
    identity === null ? "未提供身份参考图" : "ArcFace 身份证据";

  wangxingResult.classList.remove("is-hidden");
  wangxingResult.innerHTML = `
    <div class="wangxing-result-head">
      <div>
        <span class="wangxing-result-kicker">TARGET SPECIALIZATION / WANG XING AU</span>
        <h3>王兴面部表情画像评估</h3>
      </div>
      <span class="wangxing-result-status ${escapeHtml(decision)}">${escapeHtml(decisionLabel)}</span>
    </div>
    <div class="wangxing-result-score">
      <strong>${escapeHtml(scoreText)}</strong>
      <span>/100 · ${escapeHtml(classContext)}</span>
    </div>
    <p class="wangxing-result-note">
      ${escapeHtml(
        evidenceCoverage === null
          ? "AU evidence coverage unavailable"
          : `AU evidence coverage ${formatEvidence(evidenceCoverage)}; missing evidence does not mean a complete match`,
      )}
    </p>
    <div class="wangxing-result-evidence">
      <div class="wangxing-result-evidence-group">
        <span class="wangxing-result-evidence-label">面部表情证据 / FACIAL AU EVIDENCE</span>
        <div class="wangxing-result-evidence-grid">
          <span class="is-primary"><strong>${formatEvidence(personal)}</strong>王兴 AU 画像</span>
          <span class="is-primary"><strong>${formatEvidence(facialDynamics)}</strong>面部表情动态一致性</span>
          <span><strong>${escapeHtml(eventStatistics)}</strong>当前视频表情事件</span>
        </div>
      </div>
      <div class="wangxing-result-evidence-group">
        <span class="wangxing-result-evidence-label">身份证据 / IDENTITY EVIDENCE</span>
        <div class="wangxing-result-evidence-grid wangxing-result-evidence-grid-identity">
          <span><strong>${formatEvidence(leakage)}</strong>身份偏离风险</span>
          <span><strong>${formatEvidence(identity)}</strong>${escapeHtml(identityText)}</span>
        </div>
      </div>
    </div>
    <div class="wangxing-result-quality">
      <div class="wangxing-result-quality-head">
        <span>${escapeHtml(qualityLabel)}</span>
        <strong>${formatEvidence(confidence)} /100</strong>
      </div>
      <div class="wangxing-result-quality-track">
        <span style="width: ${confidence === null ? 0 : confidence * 100}%"></span>
      </div>
      <p>
        有效人脸帧 ${formatEvidence(validFrameRatio)} ·
        表情活跃帧 ${formatEvidence(eventActiveRatio)}
      </p>
    </div>
    ${renderAuTemporalEvidence(au)}
    <p class="wangxing-result-note">
      ${escapeHtml(
        decisionNote ||
          "本专项只评估面部 AU 画像和面部动态统计；不推断身体动作，也不进行训练视频时间轴对齐。参考视频不参与本专项，身份参考只用于 ArcFace 证据。",
      )}
    </p>
    <div class="wangxing-result-meta">
      evaluator ${escapeHtml(au.evaluator_version ?? "unknown")} ·
      profile ${escapeHtml(au.profile_schema_version ?? "unknown")}
    </div>
  `;
}

function renderWangxingSpecializationResult(payload) {
  const identity = payload.identity ?? {};
  const expression = payload.expression_profile ?? {};
  const identityDecision = String(identity.decision ?? "uncertain");
  const finalDecision = String(payload.decision ?? "uncertain_identity");
  const identityLabels = {
    wangxing: "王兴 / WANG XING",
    not_wangxing: "不是王兴 / NOT WANG XING",
    uncertain: "身份不确定 / UNCERTAIN",
  };
  const finalLabels = {
    wangxing_expression_compatible: "王兴，表情符合画像",
    wangxing_expression_incompatible: "王兴，但表情偏离画像",
    uncertain_identity: "身份证据不足，需要复核",
    uncertain_expression: "王兴，表情证据不足",
    not_wangxing: "不是王兴",
  };
  const percent = (value) => {
    const score = normalizeScore(value);
    return score === null ? "--" : `${(score * 100).toFixed(1)}%`;
  };
  const score = normalizeScore(identity.probability_0_1);
  const negativeProbability = normalizeScore(
    identity.negative_class_probability_0_1,
  );
  const compatibility = normalizeScore(expression.compatibility_0_1);
  const consistency = normalizeScore(identity.frame_consistency);
  const validRatio = normalizeScore(identity.valid_frame_ratio);
  const qualityWeight = normalizeScore(identity.quality_weight_mean);
  const identityStatus =
    identityDecision === "wangxing"
      ? "allow"
      : identityDecision === "not_wangxing"
        ? "block"
        : "review";
  const topProfiles = Array.isArray(expression.top_profiles)
    ? expression.top_profiles
    : [];
  const events = expression.event_statistics ?? {};
  const reasons = [
    ...(identity.uncertainty_reasons ?? []),
    ...(expression.uncertainty_reasons ?? []),
  ].filter((value, index, values) => values.indexOf(value) === index);
  const reasonText = reasons.length
    ? reasons.join(" / ")
    : "身份与表情画像证据均通过当前阈值";
  const expressionText =
    identityDecision === "wangxing"
      ? finalLabels[finalDecision] ?? finalDecision
      : finalLabels[finalDecision] ?? identityLabels[identityDecision];
  const forensics = payload.forensics ?? {};
  const forensicFusion = forensics.fusion ?? {};
  const forensicScores = forensics.scores ?? {};
  const forensicBranches = forensics.branches ?? {};
  const forensicRaw = normalizeScore(
    forensicScores.raw_real_domain_evidence_0_1 ??
      forensicFusion.raw_real_domain_evidence_0_1,
  );
  const forensicProbability = normalizeScore(
    forensicScores.calibrated_real_probability_0_1 ??
      forensicFusion.real_capture_likelihood_0_1,
  );
  const forensicFacial = normalizeScore(
    forensicBranches.facial_motion?.metrics?.raw_real_domain_evidence_0_1,
  );
  const forensicTexture = normalizeScore(
    forensicBranches.texture_detail?.metrics?.raw_real_domain_evidence_0_1,
  );
  const forensicStatus = String(
    forensics.status ?? forensicFusion.status ?? "unavailable",
  );
  const forensicWarning =
    forensicFusion.warning ??
    forensics.reason ??
    (Array.isArray(forensics.authenticity?.uncertainty_reasons)
      ? forensics.authenticity.uncertainty_reasons.join(" / ")
      : null);
  const forensicMarkup =
    Object.keys(forensics).length > 0
      ? `
        <div class="wangxing-result-evidence">
          <div class="wangxing-result-evidence-group">
            <span class="wangxing-result-evidence-label">FORENSICS / REAL VS SEEDANCE</span>
            <div class="wangxing-result-evidence-grid">
              <span class="is-primary"><strong>${escapeHtml(
                percent(forensicRaw),
              )}</strong>raw domain evidence</span>
              <span><strong>${escapeHtml(
                forensicProbability === null
                  ? "NOT CALIBRATED"
                  : percent(forensicProbability),
              )}</strong>calibrated real probability</span>
              <span><strong>${escapeHtml(
                percent(forensicFacial),
              )}</strong>facial motion branch</span>
              <span><strong>${escapeHtml(
                percent(forensicTexture),
              )}</strong>texture branch</span>
            </div>
          </div>
          <p class="wangxing-result-note">
            status ${escapeHtml(forensicStatus)}${forensicWarning ? ` / ${escapeHtml(
              forensicWarning,
            )}` : ""}
          </p>
        </div>
      `
      : "";

  wangxingResult.classList.remove("is-hidden");
  wangxingResult.innerHTML = `
    <div class="wangxing-result-head">
      <div>
        <span class="wangxing-result-kicker">TARGET SPECIALIZATION / WANG XING</span>
        <h3>王兴身份与面部表情画像</h3>
      </div>
      <span class="wangxing-result-status ${escapeHtml(identityStatus)}">
        ${escapeHtml(identityLabels[identityDecision] ?? "UNCERTAIN")}
      </span>
    </div>
    <div class="wangxing-result-score">
      <strong>${escapeHtml(expressionText)}</strong>
      <span>串联判断 / IDENTITY → EXPRESSION</span>
    </div>
    <div class="wangxing-result-evidence">
      <div class="wangxing-result-evidence-group">
        <span class="wangxing-result-evidence-label">人物身份 / IDENTITY</span>
        <div class="wangxing-result-evidence-grid wangxing-result-evidence-grid-identity">
          <span class="is-primary"><strong>${escapeHtml(percent(score))}</strong>王兴身份概率</span>
          <span><strong>${escapeHtml(percent(negativeProbability))}</strong>负样本分类概率</span>
          <span><strong>${escapeHtml(percent(consistency))}</strong>人脸帧一致性</span>
          <span><strong>${escapeHtml(percent(validRatio))}</strong>有效人脸帧</span>
          <span><strong>${escapeHtml(percent(qualityWeight))}</strong>平均帧质量权重</span>
          <span><strong>${escapeHtml(String(identity.valid_frame_count ?? "--"))}</strong>有效帧数</span>
        </div>
      </div>
      <div class="wangxing-result-evidence-group">
        <span class="wangxing-result-evidence-label">表情画像 / EXPRESSION SUPPORT DOMAIN</span>
        <div class="wangxing-result-evidence-grid">
          <span class="is-primary"><strong>${escapeHtml(percent(compatibility))}</strong>画像符合度</span>
          <span><strong>${escapeHtml(expression.selected_profile_display_name ?? "--")}</strong>最接近画像</span>
          <span><strong>${escapeHtml(percent(expression.margin_0_1))}</strong>前二画像间隔</span>
          <span><strong>${expression.severe_deviation ? "是" : "否"}</strong>严重偏离</span>
        </div>
      </div>
    </div>
    ${
      topProfiles.length
        ? `<div class="wangxing-result-note">
            最接近的两个画像：
            ${topProfiles
              .map(
                (profile, index) =>
                  `${index + 1}. ${escapeHtml(
                    profile.display_name ?? profile.class ?? "--",
                  )} ${escapeHtml(percent(profile.score_0_1))}`,
              )
              .join(" / ")}
          </div>`
        : ""
    }
    <div class="wangxing-result-quality">
      <div class="wangxing-result-quality-head">
        <span>当前视频表情事件</span>
        <strong>${escapeHtml(String(events.event_count ?? "--"))} 次</strong>
      </div>
      <div class="wangxing-result-quality-track">
        <span style="width: ${Math.max(
          0,
          Math.min(100, Number(events.active_ratio ?? 0) * 100),
        )}%"></span>
      </div>
      <p>
        激活比例 ${escapeHtml(percent(events.active_ratio))} /
        最长事件 ${escapeHtml(percent(events.longest_event_ratio))} /
        峰值 ${escapeHtml(String(events.peak_intensity ?? "--"))}
      </p>
    </div>
    ${forensicMarkup}
    <p class="wangxing-result-note">
      ${escapeHtml(reasonText)}
    </p>
    <div class="wangxing-result-meta">
      evaluator ${escapeHtml(payload.evaluator_version ?? "unknown")} /
      identity ${escapeHtml(identity.backend ?? "unknown")} /
      expression ${escapeHtml(expression.selected_profile ?? "not evaluated")}
    </div>
  `;
}

function renderDownloads(downloads) {
  const links = [
    ["summary_csv", "summary.csv"],
    ["frame_csv", "frame_metrics.csv"],
    ["result_json", "result.json"],
    ["wangxing_au_json", "wangxing_au_result.json"],
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
  const scoreStatusLabels = {
    complete: "完整评估",
    partial: "部分评估",
  };
  const coveragePercent = result.weighted_score_weight_coverage ?? 0;
  const scoreStatus = scoreStatusLabels[result.status] ?? result.status ?? "待确认";
  const hardware = result.hardware_policy ?? {};
  const deviceLabels = { cpu: "CPU", cuda: "CUDA" };
  const actualDevice = deviceLabels[hardware.resolved_device] ?? "";
  const requestedDevice = deviceLabels[hardware.requested_device] ?? "";
  const deviceNote = actualDevice
    ? ` · 实际设备 ${actualDevice}${
        requestedDevice && requestedDevice !== actualDevice
          ? `（请求 ${requestedDevice}）`
          : ""
      }`
    : "";
  scoreCaption.textContent = `权重覆盖 ${coveragePercent}% · ${scoreStatus}${deviceNote}`;
  const covered = Number.parseInt(coverage, 10) || 0;
  coverageRing.style.setProperty("--coverage", `${(covered / 5) * 100}%`);
  renderRadar(result);
  renderWangxingResult(result);
  renderCategories(result);
  renderQwenFeedback(result);
  renderEvidence(result);
  renderDownloads(payload.downloads);
  document.querySelector("#report-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderEmptyReport(status = "queued", errorMessage = "") {
  const emptyTitle = emptyReport.querySelector("h3");
  const emptyDescription = emptyReport.querySelector(
    ".empty-report-head > div > p:last-child",
  );
  const emptySignal = emptyReport.querySelector(".empty-signal");
  if (status === "failed") {
    emptyTitle.textContent = "本次评估失败。";
    emptyDescription.textContent =
      errorMessage || "请检查输入文件、模型服务和显存状态后重试。";
    emptySignal.textContent = "EVALUATION FAILED";
  } else if (status === "canceled") {
    emptyTitle.textContent = "本次任务已中断。";
    emptyDescription.textContent =
      "可以直接修改左侧参数并重新评估，已上传素材会继续复用。";
    emptySignal.textContent = "TASK CANCELED";
  } else {
    emptyTitle.textContent = "上传视频后开始评分。";
    emptyDescription.textContent =
      "系统会根据你提供的 GT、参考图、参考视频和 Prompt，自动选择可用指标。";
    emptySignal.textContent = "WAITING FOR VIDEO";
  }
  emptyReport.classList.remove("is-hidden");
  reportContent.classList.add("is-hidden");
  reportMode.textContent =
    status === "failed"
      ? "FAILED"
      : status === "canceled"
        ? "CANCELED"
        : "WAITING";
  reportMode.classList.remove("success");
  overallScore.textContent = "--";
  scoreCaption.textContent = "覆盖情况";
  coverageValue.textContent = "0/5";
  coverageRing.style.setProperty("--coverage", "0%");
  document.querySelector("#radar-chart").innerHTML = "";
  categoryList.innerHTML = "";
  evidenceGrid.innerHTML = "";
  downloadRow.innerHTML = "";
  wangxingResult.classList.add("is-hidden");
  wangxingResult.innerHTML = "";
  qwenFeedback.classList.add("is-hidden");
  qwenFeedback.innerHTML = "";
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

loadModels();
window.setInterval(loadModels, 15_000);

window.queueMode = true;

const queueStageLabels = {
  queued: ["upload", "排队等待"],
  preparing: ["upload", "准备输入文件"],
  sampling: ["sample", "抽取关键帧"],
  models: ["models", "运行本地模型"],
  wangxing_au: ["models", "王兴面部表情画像评估"],
  report: ["report", "整理评估报告"],
  completed: ["report", "评估完成"],
  failed: ["report", "评估失败"],
  canceled: ["upload", "任务已取消"],
  canceling: ["upload", "正在中断任务"],
};
const queueStatusLabels = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  canceled: "已取消",
  canceling: "中断中",
};
const queueTerminalStatuses = new Set(["completed", "failed", "canceled"]);
const activeQueueStatuses = new Set(["running", "canceling"]);
const retryableStatuses = new Set(["failed", "canceled", "completed"]);
const formLockedStatuses = new Set([
  "queued",
  "running",
  "canceling",
]);
const queueKnownStatuses = new Map();
let selectedJobId = null;
let selectedJob = null;
let formLocked = false;
let formBusy = false;
let queueRefreshInFlight = false;
let queueMutationJobId = null;
let latestQueuePayload = null;

function setQueueBusy(isBusy) {
  formBusy = isBusy;
  evaluateButton.disabled = isBusy;
  newEvaluationButton.disabled = isBusy;
  evaluateButton.querySelector("span:first-child").textContent = isBusy
    ? "上传中..."
    : retryableStatuses.has(selectedJob?.status)
      ? "重新评估"
      : "加入队列";
}

function setFormEditState(job) {
  formLocked = Boolean(job && formLockedStatuses.has(job.status));
  const canReuseStoredFiles = Boolean(
    job &&
      retryableStatuses.has(job.status) &&
      job.uploaded_files?.result_video,
  );
  form.classList.toggle("is-readonly", formLocked);
  newEvaluationButton.disabled = formBusy;
  form.querySelectorAll("input, textarea, select").forEach((field) => {
    field.disabled = formLocked;
  });
  const resultInput = document.querySelector("#result-video");
  if (resultInput) {
    resultInput.required = !canReuseStoredFiles;
  }
  evaluateButton.disabled = formBusy || formLocked;
  if (!formBusy) {
    evaluateButton.querySelector("span:first-child").textContent = formLocked
      ? "请先中断任务"
      : canReuseStoredFiles
        ? "重新评估"
        : "加入队列";
  }
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

function formatQueueTimestamp(value) {
  const timestamp = Date.parse(value ?? "");
  if (!Number.isFinite(timestamp)) return "时间未知";
  const parts = new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(new Date(timestamp));
  const values = Object.fromEntries(
    parts
      .filter(({ type }) => type !== "literal")
      .map(({ type, value }) => [type, value]),
  );
  return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second}`;
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "时间未知";
  if (value < 60) return `${Math.max(1, Math.round(value))} 秒`;
  const minutes = Math.floor(value / 60);
  const remainder = Math.round(value % 60);
  return remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分钟`;
}

function queueItemTimestamp(job) {
  return `评分时间 ${formatQueueTimestamp(
    job.finished_at || job.started_at || job.created_at || job.updated_at,
  )}`;
}

function renderQueue(payload) {
  latestQueuePayload = payload;
  const jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
  const searchTerm = String(queueSearch?.value ?? "").trim().toLowerCase();
  const filteredJobs = searchTerm
    ? jobs.filter((job) =>
        [job.name, job.job_id, job.status, job.error]
          .filter(Boolean)
          .concat(queueStatusText(job.status))
          .some((value) => String(value).toLowerCase().includes(searchTerm)),
      )
    : jobs;
  const active = jobs.find((job) => activeQueueStatuses.has(job.status));
  const activeRunning = active?.status === "running";
  queueSummary.textContent = `${activeRunning ? 1 : 0} 个运行 / ${
    payload.queued_count ?? 0
  } 个等待`;
  queueActive.classList.toggle("is-hidden", !active);
  if (active) {
    activeJobName.textContent = active.name || active.job_id;
    activeJobStage.textContent =
      queueStageLabels[active.stage]?.[1] ?? active.stage;
    const startedAt = Date.parse(
      active.started_at || active.created_at || active.updated_at || "",
    );
    const elapsed = Number.isFinite(startedAt)
      ? Math.max(0, (Date.now() - startedAt) / 1000)
      : 0;
    const remaining = Math.max(
      0,
      Number(active.estimated_seconds ?? 0) - elapsed,
    );
    activeJobTime.textContent = `开始时间 ${formatQueueTimestamp(
      active.started_at || active.created_at || active.updated_at,
    )} · 预计还需 ${formatDuration(remaining)}`;
    activeJobStatus.textContent = queueStatusText(active.status);
    activeJobStatus.className = `queue-status ${escapeHtml(active.status)}`;
    activeJobCancel.dataset.jobId = active.job_id;
    activeJobCancel.textContent =
      active.status === "canceling" ? "中断中..." : "中断任务";
    activeJobCancel.disabled =
      active.status === "canceling" || queueMutationJobId === active.job_id;
    queueActive.dataset.jobId = active.job_id;
    queueActive.classList.add("is-selectable");
    activeJobProgress.style.width = `${Math.max(
      0,
      Math.min(100, Number(active.progress ?? 0) * 100),
    )}%`;
  } else {
    activeJobCancel.dataset.jobId = "";
    activeJobCancel.disabled = true;
    activeJobCancel.textContent = "中断任务";
    activeJobTime.textContent = "评分时间 --";
    queueActive.dataset.jobId = "";
    queueActive.classList.remove("is-selectable");
  }

  const visibleJobs = filteredJobs
    .filter((job) => !activeQueueStatuses.has(job.status));
  queueList.innerHTML = visibleJobs
    .map((job) => {
      const position =
        job.status === "queued" && job.queue_position != null
          ? String(job.queue_position).padStart(2, "0")
          : "·";
      const actions = [];
      if (job.status === "queued") {
        actions.push(["cancel", "取消"]);
      }
      if (retryableStatuses.has(job.status)) {
        actions.push(["retry", "重试"]);
      }
      if (job.status !== "running" && job.status !== "canceling") {
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
          <span class="queue-item-index">${position}</span>
          <button class="queue-item" type="button" data-job-id="${escapeHtml(job.job_id)}">
            <span class="queue-item-copy">
              <strong>${escapeHtml(job.name || job.job_id)}</strong>
              <span class="queue-item-meta">
                <small class="queue-item-stage">${escapeHtml(queueStageLabels[job.stage]?.[1] ?? queueStatusText(job.status))}</small>
                <small class="queue-item-time">${escapeHtml(queueItemTimestamp(job))}</small>
              </span>
            </span>
            <span class="queue-item-status ${escapeHtml(job.status)}">${queueStatusText(job.status)}</span>
          </button>
          <div class="queue-item-actions">${actionMarkup}</div>
        </div>
      `;
    })
    .join("");
  queueEmpty.textContent = searchTerm
    ? "没有匹配的任务"
    : "队列为空 / 上传视频后开始";
  queueEmpty.classList.toggle("is-hidden", filteredJobs.length > 0);

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

function setStoredUpload(inputId, filename, url) {
  const input = document.querySelector(`#${inputId}`);
  const zone = input?.closest("label");
  const name = document.querySelector(`[data-file-name="${inputId}"]`);
  const preview = document.querySelector(`[data-preview="${inputId}"]`);
  const fallback = document.querySelector(`[data-preview-fallback="${inputId}"]`);
  const imageList = document.querySelector(`[data-image-list="${inputId}"]`);
  if (!input || !zone || !name) return;

  input.value = "";
  if (!filename) {
    zone.classList.remove("is-loaded", "video-preview-failed");
    name.textContent = "未提供";
    if (imageList) imageList.innerHTML = "";
    if (preview) preview.removeAttribute("src");
    return;
  }

  zone.classList.add("is-loaded");
  zone.classList.remove("video-preview-failed");
  if (fallback) fallback.classList.remove("is-visible");
  if (Array.isArray(filename) && inputId === "reference-video") {
    name.textContent = `已保存：${filename.length} 段参考视频`;
  } else {
    name.textContent = `已保存：${filename}`;
  }

  if (preview && url) {
    preview.src = url;
    preview.load();
  }
  if (imageList) {
    imageList.innerHTML = "";
    const urls = Array.isArray(url) ? url : url ? [url] : [];
    urls.forEach((imageUrl, index) => {
      const image = document.createElement("img");
      image.src = imageUrl;
      image.alt = Array.isArray(filename) ? filename[index] ?? "" : filename;
      imageList.appendChild(image);
    });
  }
}

function clearUploadState(inputId, defaultName) {
  const input = document.querySelector(`#${inputId}`);
  const zone = input?.closest("label");
  const name = document.querySelector(`[data-file-name="${inputId}"]`);
  const preview = document.querySelector(`[data-preview="${inputId}"]`);
  const imageList = document.querySelector(`[data-image-list="${inputId}"]`);
  if (!input || !zone || !name) return;

  for (const [key, url] of previewUrls.entries()) {
    if (key === inputId || key.startsWith(`${inputId}:`)) {
      URL.revokeObjectURL(url);
      previewUrls.delete(key);
    }
  }
  input.value = "";
  zone.classList.remove("is-loaded", "video-preview-failed");
  name.textContent = defaultName;
  if (imageList) imageList.innerHTML = "";
  if (preview) {
    preview.pause();
    preview.removeAttribute("src");
    preview.load();
  }
}

function startNewEvaluation() {
  if (newEvaluationButton.disabled) return;
  selectedJobId = null;
  selectedJob = null;
  setFormEditState(null);
  form.reset();
  clearUploadState("result-video", "Drop video or browse");
  clearUploadState("gt-video", "Optional reference");
  clearUploadState("reference-images", "optional");
  clearUploadState("reference-video", "optional");
  document.querySelector("#result-video").required = true;
  updateQueueProgressPanel(null);
  progressBar.classList.remove("is-complete");
  progressBar.style.width = "0%";
  renderEmptyReport();
  setQueueBusy(false);
  setFormNote("已清空当前表单，可以开始新的评估。", "success");
  document.querySelector(".intake-panel").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

function syncFormWithJob(job) {
  const parameters = job.parameters ?? {};
  const originalFiles = job.original_files ?? {};
  const uploadedFiles = job.uploaded_files ?? {};
  const fileNameFromUrl = (url) =>
    String(url ?? "").split("/").pop() || "";
  const referenceImageNames =
    originalFiles.reference_images?.join("、") ||
    (Array.isArray(uploadedFiles.reference_images)
      ? uploadedFiles.reference_images.map(fileNameFromUrl).join("、")
      : "");
  const setValue = (selector, value) => {
    const field = document.querySelector(selector);
    if (field && value !== null && value !== undefined) {
      field.value = value;
    }
  };

  setValue("#evaluation-name", job.name ?? "");
  setValue("#prompt-text", parameters.prompt_text ?? "");
  setValue('[name="max_frames"]', parameters.max_frames ?? 8);
  setValue('[name="device"]', parameters.device ?? "cuda");
  setValue(
    '[name="manual_expression_score"]',
    parameters.manual_expression_score ?? "",
  );
  setValue(
    '[name="manual_aesthetic_score"]',
    parameters.manual_aesthetic_score ?? "",
  );
  setValue(
    '[name="wangxing_expected_class"]',
    parameters.wangxing_expected_class ?? "auto",
  );
  const lpips = document.querySelector('[name="calculate_lpips"]');
  if (lpips) lpips.checked = parameters.calculate_lpips !== false;
  const wangxingAu = document.querySelector('[name="wangxing_au_enabled"]');
  if (wangxingAu) {
    wangxingAu.checked = parameters.wangxing_au_enabled === true;
  }

  setStoredUpload(
    "result-video",
    originalFiles.result_video ?? job.name,
    uploadedFiles.result_video,
  );
  setStoredUpload(
    "gt-video",
    originalFiles.gt_video,
    uploadedFiles.gt_video,
  );
  setStoredUpload(
    "reference-images",
    referenceImageNames,
    uploadedFiles.reference_images,
  );
  setStoredUpload(
    "reference-video",
    originalFiles.reference_video,
    uploadedFiles.reference_video,
  );
}

async function getQueueJob(jobId, shouldScroll = false) {
  if (!jobId) return null;
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(describeApiError(payload.detail, response.status));
  }
  selectedJobId = payload.job_id;
  selectedJob = payload;
  syncFormWithJob(payload);
  setFormEditState(payload);
  updateQueueProgressPanel(payload);
  if (payload.result) {
    renderResult(payload);
    if (shouldScroll) {
      document.querySelector("#report-panel").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  } else {
    renderEmptyReport(payload.status, payload.error);
  }
  return payload;
}

async function selectQueueJob(jobId) {
  try {
    const payload = await getQueueJob(jobId, true);
    if (payload?.status === "running") {
      setFormNote("已加载运行中任务，当前仅可查看；请先中断后修改。");
    } else if (payload?.status === "canceling") {
      setFormNote("任务正在中断，暂时无法修改。");
    } else if (payload?.status === "canceled") {
      setFormNote("任务已中断，可以修改左侧参数后直接重新评估。", "success");
    } else if (payload?.status === "failed") {
      setFormNote("任务失败，可以修改左侧参数后重新评估。", "error");
    } else if (payload?.status === "completed") {
      setFormNote(
        `已加载报告，可以修改左侧参数后重新评估 / ${payload.name}`,
        "success",
      );
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
    action === "cancel" &&
    !window.confirm("确定中断这个评估任务吗？已完成的部分不会生成报告。")
  ) {
    return;
  }
  if (
    action === "delete" &&
    !window.confirm("确定删除这个任务及其已保存的文件吗？")
  ) {
    return;
  }
  if (action === "cancel" && jobId === activeJobCancel.dataset.jobId) {
    queueMutationJobId = jobId;
    activeJobCancel.disabled = true;
    activeJobCancel.textContent = "中断中...";
    setFormNote("正在中断评估任务...");
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
      startNewEvaluation();
    }
    if (selectedJobId === jobId && action === "cancel") {
      selectedJob = payload;
      syncFormWithJob(payload);
      setFormEditState(payload);
    }
    await refreshQueue();
  } catch (error) {
    setFormNote(error.message || "无法更新队列任务。", "error");
  } finally {
    if (queueMutationJobId === jobId) {
      queueMutationJobId = null;
    }
  }
}

async function refreshQueue() {
  if (queueRefreshInFlight) return;
  queueRefreshInFlight = true;
  try {
    const response = await fetch("/api/jobs?limit=100");
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(describeApiError(payload.detail, response.status));
    }
    renderQueue(payload);
    const selected = payload.jobs?.find(
      (job) => job.job_id === selectedJobId,
    );
    if (selected) {
      selectedJob = selected;
      setFormEditState(selected);
      updateQueueProgressPanel(selected);
      if (selected.status !== "completed") {
        renderEmptyReport(selected.status, selected.error);
      }
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
queueSearch?.addEventListener("input", () => {
  if (latestQueuePayload) renderQueue(latestQueuePayload);
});
newEvaluationButton.addEventListener("click", startNewEvaluation);
queueActive.addEventListener("click", (event) => {
  if (event.target.closest("#active-job-cancel")) return;
  if (queueActive.dataset.jobId) {
    selectQueueJob(queueActive.dataset.jobId);
  }
});
activeJobCancel.addEventListener("click", () => {
  mutateQueueJob(activeJobCancel.dataset.jobId, "cancel");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (formLocked) {
    setFormNote("当前任务仅供查看，请先中断任务后再修改。");
    return;
  }
  setQueueBusy(true);
  const resultInput = document.querySelector("#result-video");
  const hasNewUploads = Array.from(
    form.querySelectorAll('input[type="file"]'),
  ).some((input) => input.files?.length > 0);
  const hasNewResult = Boolean(resultInput?.files?.length);
  const canReuseStoredFiles = Boolean(
    selectedJob &&
      retryableStatuses.has(selectedJob.status) &&
      selectedJob.uploaded_files?.result_video &&
      !hasNewUploads,
  );
  const canReuseStoredOptionalFiles = Boolean(
    selectedJob &&
      selectedJobId &&
      retryableStatuses.has(selectedJob.status) &&
      hasNewResult,
  );
  if (hasNewUploads && !hasNewResult) {
    setQueueBusy(false);
    setFormNote(
      "如果要更换参考素材，请同时重新选择结果视频；否则这些新文件不会进入本次评估。",
      "error",
    );
    return;
  }
  setFormNote(
    canReuseStoredFiles
      ? "正在使用已保存素材并重新加入处理队列..."
      : canReuseStoredOptionalFiles
        ? "正在上传新结果视频并复用已保存参考素材..."
        : "正在上传文件并加入处理队列...",
  );
  try {
    const formData = new FormData(form);
    formData.set(
      "calculate_lpips",
      form.querySelector('[name="calculate_lpips"]').checked ? "true" : "false",
    );
    formData.set(
      "wangxing_au_enabled",
      form.querySelector('[name="wangxing_au_enabled"]').checked ? "true" : "false",
    );
    if (canReuseStoredOptionalFiles) {
      formData.set("reuse_job_id", selectedJobId);
    } else {
      formData.delete("reuse_job_id");
    }
    let response;
    if (canReuseStoredFiles && selectedJobId) {
      const optionalNumber = (name) => {
        const value = formData.get(name);
        return value === null || String(value).trim() === ""
          ? null
          : Number(value);
      };
      response = await fetch(
        `/api/jobs/${encodeURIComponent(selectedJobId)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "retry",
            name: String(formData.get("name") ?? "").trim(),
            prompt_text: String(formData.get("prompt_text") ?? ""),
            max_frames: Number(formData.get("max_frames") ?? 8),
            calculate_lpips: formData.get("calculate_lpips") === "true",
            device: String(formData.get("device") ?? "auto"),
            manual_expression_score: optionalNumber("manual_expression_score"),
            manual_aesthetic_score: optionalNumber("manual_aesthetic_score"),
            wangxing_au_enabled:
              formData.get("wangxing_au_enabled") === "true",
            wangxing_expected_class: String(
              formData.get("wangxing_expected_class") ?? "auto",
            ),
          }),
        },
      );
    } else {
      response = await fetch("/api/jobs", {
        method: "POST",
        body: formData,
      });
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(describeApiError(payload.detail, response.status));
    }
    if (!payload.job_id) {
      throw new Error("队列没有返回任务编号。");
    }
    selectedJobId = payload.job_id;
    selectedJob = payload;
    queueKnownStatuses.set(payload.job_id, payload.status);
    setFormEditState(payload);
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
