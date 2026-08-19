# Human Review

独立的视频生成结果人工二选一评测页。用户只需要判断：

> 哪个视频中的人物表演更像真人？

选项为：

- `A`
- `B`
- `一样 / 无法评估`

## 启动

在项目根目录运行：

```powershell
python human_review/server.py --host 0.0.0.0 --port 5001
```

浏览器打开 `http://127.0.0.1:5001`。

默认监听所有网卡，因此同一局域网内的其他设备可以访问：

```text
http://<运行服务电脑的局域网IP>:5001
```

也可以直接在 `human_review` 目录运行：

```powershell
python server.py --port 5001
```

如果只允许本机访问，显式指定：

```powershell
python human_review/server.py --host 127.0.0.1 --port 5001
```

该服务默认没有登录鉴权，不要直接暴露到公网；公网部署应放在带鉴权、
HTTPS 和访问控制的反向代理后面。

## 数据

运行时任务来自 SQLite 数据库。原始实验数据可以用构建器整理成版本化数据集：

```powershell
python human_review/build_dataset.py `
  --raw-root human_review\data\raw_archive\experiments_20260811 `
  --output-dir human_review\data\datasets\performance_v8 `
  --db human_review\data\review.sqlite3 `
  --dataset-id performance_v8 `
  --per-reviewer-quota 80 `
  --max-video-seconds 10 `
  --max-video-width 720
```

构建器会输出 `dataset.json`、`assets.jsonl`、`tasks.jsonl` 和跳过批次清单，
同时把任务和媒体索引写入 SQLite。原始素材不会被移动或复制。

任务类型由任务自身字段决定，不由前端猜测：

- `ai_real_anchor`：只判断人物表演的真人感；投票后揭示 `AI 生成 / 实拍`。
- `model_comparison`：只接收同一案例、同一 Prompt、同一参考内容下的明确跨模型配对；
  投票前显示 Prompt 和参考素材，投票后揭示具体模型名。
- `control`：重复视频控制题，不参与模型排名。

无法证明比较条件一致的跨模型候选会写入 `skipped_batches.json` 的
`cross_model_pair_requires_manual_review`，不会自动进入投票。

示例：

```json
{
  "task_id": "task_00018",
  "modality": "multi_reference",
  "prompt": "一个人在山顶露营，镜头缓慢推进",
  "references": [
    {
      "type": "image",
      "path": "references/camp.jpg"
    }
  ],
  "candidates": [
    {
      "candidate_id": "candidate_x",
      "model_id": "private_model_x",
      "video_path": "candidates/task_00018_x.mp4"
    },
    {
      "candidate_id": "candidate_y",
      "model_id": "private_model_y",
      "video_path": "candidates/task_00018_y.mp4"
    }
  ]
}
```

候选必须严格为两个。`model_id` 不会返回给浏览器，只在后端任务清单和投票映射
中保存。

投票写入 `data/review.sqlite3`，包含：

- 当前会话看到的 A/B 候选映射
- 选择结果
- 响应时长
- 任务 ID 和时间
- 浏览器评测身份的 HMAC 哈希
- IP 的 HMAC 哈希审计字段

同一个 `dataset_id + task_id + reviewer_id_hash` 只能写入一次。
`round_id` 只用于记录浏览器轮次；A/B 展示顺序固定绑定浏览器身份，
不能通过新轮次绕过重复评分限制。

每个数据集可以配置 `per_reviewer_quota`。当前 `performance_v8` 有 58 道可投票题，
旧数据库中的 `per_ip_quota` 仅作为兼容字段，不再按 IP 限制不同用户。
其中包含 1 道 LTX2.3 vs Seedance 2.0；控制题和 1 道
`needs_manual_review` 记录不会被服务分发。

当前服务只使用 `performance_v8` 数据集和
`human_review/data/review.sqlite3` 数据库。若要切换数据库或数据集，
通过 `HUMAN_REVIEW_DB` 和 `HUMAN_REVIEW_DATASET` 显式配置。

## AI 视频质量分档栏目

页面顶部另有独立的 `AI 质量分档` 栏目。它不改变现有
`performance_v8` 二选一任务，使用同一个浏览器身份和 IP 审计哈希，
但任务、媒体和评分分别保存在 `quality_*` 表中。

当前初始化数据集是 25+5 条 AI 视频：

```text
human_review/data/ai_quality/videos/
human_review/data/ai_quality/manifest.json
human_review/data/datasets/ai_quality_25plus5_v1/
```

页面每次显示一个 AI 视频，评分按钮为 `上档`、`中档`、`下档`。
程序分数和已有人工分档只保存在服务端数据中，不发送到网页，避免影响
新的主观评分。该栏目不限制视频时长，完整视频可通过播放器观看。
每个版本构建时都会保存独立的视频快照，避免后续替换输入文件影响旧版本。

构建或更新数据集：

```powershell
python human_review/build_ai_quality_dataset.py
```

已有评分的数据集不能原地重建。添加新视频后应使用新版本：

```powershell
python human_review/build_ai_quality_dataset.py `
  --dataset-id ai_quality_25plus5_v2 `
  --output-dir human_review\data\datasets\ai_quality_25plus5_v2
$env:HUMAN_REVIEW_QUALITY_DATASET = "ai_quality_25plus5_v2"
```

新栏目接口：

```text
GET  /api/quality/next
GET  /api/quality/progress
POST /api/quality/rate
GET  /api/quality/health
GET  /media/quality/<dataset-id>/<asset-id>
```

导出 AI 分档统计：

```powershell
python human_review/export_ai_quality_results.py `
  --dataset-id ai_quality_25plus5_v1 `
  --csv human_review\data\reports\ai_quality_25plus5_v1_summary.csv
```

导出结果包含每条视频的三档计数、主档位、一致率，以及原程序分档和表情分，
不会包含 reviewer 或 IP 哈希。

## 带参考生成实验

原始实验只保存在 `human_review/data/raw_archive/experiments_20260811`。
可读导出在 `data/test/with_reference`。

```powershell
python scripts/export_human_review_reference_set.py
```

导出结构：

```text
with_reference/experiments/exp_001__helmet_identity_views/
  prompt.txt
  reference_inputs/images|audio|videos
  generated_videos/run_01__seedance_2_0.mp4
  experiment.json
```

如果服务位于可信反向代理之后，再显式开启代理 IP 读取：

```powershell
$env:HUMAN_REVIEW_TRUST_PROXY_HEADERS = "true"
$env:HUMAN_REVIEW_IP_SECRET = "replace-with-a-stable-secret"
python human_review/server.py --port 5001
```

不要在不可信代理环境中开启 `HUMAN_REVIEW_TRUST_PROXY_HEADERS`。

服务通过 HttpOnly Cookie 为每个浏览器分配独立评测身份。同一局域网内的不同
浏览器拥有不同进度和评分记录；IP 只保存为不可逆审计哈希。清除 Cookie、
更换浏览器或使用隐私窗口会被视为新的评测身份，因此正式统计应通过登录、
邀请链接或上游访问控制限制身份数量。使用 HTTPS 部署时，设置
`HUMAN_REVIEW_COOKIE_SECURE=true`。

## 接口

```text
GET  /api/review/next
GET  /api/review/progress
POST /api/review/vote
GET  /api/review/health
GET  /media/asset/<dataset-id>/<asset-id>
```

投票请求：

```json
{
  "task_id": "task_00018",
  "choice": "A",
  "response_ms": 4210
}
```

`choice` 只能是 `A`、`B` 或 `tie_or_unrateable`。
