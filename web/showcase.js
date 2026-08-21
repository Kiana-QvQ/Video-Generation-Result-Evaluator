const queueNode = document.querySelector("#queue");
const detailNode = document.querySelector("#detail");
const searchNode = document.querySelector("#search");
const statusNode = document.querySelector("#status");
let items = [];
let selectedId = "";

function text(value, fallback = "--") {
  return value === null || value === undefined || value === ""
    ? fallback
    : String(value);
}

function escapeHtml(value) {
  return text(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
}

function binaryConclusion(value, fallback = "偏向 AI 生成") {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  const probability = number <= 1 ? number : number / 100;
  return probability >= 0.5 ? "偏向真实拍摄" : "偏向 AI 生成";
}

function preview(item) {
  const value = item.preview || {};
  return {
    title: value.title || item.title,
    conclusion: value.conclusion || value.status || item.label,
    real: value.real_probability,
    identity: value.identity_score,
    expression: value.expression_score,
    forensics: value.forensics_score,
  };
}

function renderQueue() {
  const query = String(searchNode.value || "").trim().toLowerCase();
  const visible = items.filter((item) => {
    if (!query) return true;
    return [item.title, item.category, item.sample_id, item.label]
      .map((value) => String(value || "").toLowerCase())
      .some((value) => value.includes(query));
  });
  queueNode.innerHTML = "";
  statusNode.textContent = `${visible.length} 条结果 / 共 ${items.length} 条`;
  if (!visible.length) {
    queueNode.innerHTML = '<p class="empty">没有匹配的结果。</p>';
    return;
  }
  for (const item of visible) {
    const button = document.createElement("button");
    button.className = `queue-item${item.item_id === selectedId ? " active" : ""}`;
    button.type = "button";
    button.innerHTML = `
      <strong>${escapeHtml(item.title)}</strong>
      <small>${escapeHtml(item.category)} · ${escapeHtml(item.sample_id)}</small>
      <span class="tag">${escapeHtml(item.label)}</span>
    `;
    button.addEventListener("click", () => loadDetail(item.item_id));
    queueNode.appendChild(button);
  }
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderDetail(payload) {
  const item = payload.item || {};
  const result = payload.result || {};
  const card = result.web_card || {};
  const previewValue = preview(item);
  const forensics = result.forensics || {};
  const wangxing = result.wangxing_au || result.wangxing || {};
  const rawIdentity = wangxing.identity || {};
  const expression = wangxing.expression_profile || {};
  const probability =
    previewValue.real ??
    card.forensics?.calibrated_real_probability ??
    forensics.scores?.calibrated_real_probability_0_1;
  const identity =
    previewValue.identity ??
    card.radar?.identity?.score ??
    rawIdentity.probability_0_1;
  const expressionScore =
    previewValue.expression ??
    card.radar?.expression?.score ??
    expression.compatibility_0_1;
  const forensicsScore =
    previewValue.forensics ??
    card.radar?.forensics?.score ??
    forensics.scores?.raw_real_domain_evidence_0_1;
  const conclusion = binaryConclusion(
    probability,
    previewValue.conclusion ||
      card.forensics?.conclusion ||
      forensics.summary?.conclusion ||
      item.label,
  );
  const links = Object.entries(payload.downloads || {})
    .map(
      ([key, url]) =>
        `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(key)}</a>`,
    )
    .join(" · ");
  detailNode.innerHTML = `
    <h2>${escapeHtml(item.title)}</h2>
    <p>${escapeHtml(item.category)} · ${escapeHtml(item.sample_id)} · ${escapeHtml(item.published_at)}</p>
    <div class="headline">
      ${metric(escapeHtml("真实性概率"), escapeHtml(percent(probability)))}
      ${metric(escapeHtml("身份证据"), escapeHtml(percent(identity)))}
      ${metric(escapeHtml("表情证据"), escapeHtml(percent(expressionScore)))}
      ${metric(escapeHtml("取证证据"), escapeHtml(percent(forensicsScore)))}
    </div>
    <div class="conclusion"><strong>${escapeHtml(conclusion)}</strong><br>
      <span>身份、表情、取证分数是不同证据维度，不等同于真实拍摄概率。</span>
    </div>
    <div class="links">${links || "暂无下载文件"}</div>
    <details>
      <summary>查看完整 JSON</summary>
      <pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre>
    </details>
  `;
}

async function loadDetail(itemId) {
  selectedId = itemId;
  renderQueue();
  const response = await fetch(`/api/public-showcase/${encodeURIComponent(itemId)}`);
  if (!response.ok) {
    detailNode.textContent = `读取失败：${response.status}`;
    return;
  }
  renderDetail(await response.json());
}

async function loadQueue() {
  const response = await fetch("/api/public-showcase?limit=1000");
  if (!response.ok) {
    statusNode.textContent = `公共队列未就绪：${response.status}`;
    return;
  }
  const payload = await response.json();
  items = Array.isArray(payload.items) ? payload.items : [];
  renderQueue();
  if (items.length && !selectedId) loadDetail(items[0].item_id);
}

searchNode.addEventListener("input", renderQueue);
loadQueue();
window.setInterval(loadQueue, 15000);
