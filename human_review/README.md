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

也可以直接在 `human_review` 目录运行：

```powershell
python server.py --port 5001
```

## 数据

运行时任务来自 SQLite 数据库。原始实验数据可以用构建器整理成版本化数据集：

```powershell
python human_review/build_dataset.py `
  --raw-root human_review\data\raw_archive\experiments_20260811 `
  --output-dir human_review\data\datasets\performance_v6 `
  --db human_review\data\review.sqlite3 `
  --dataset-id performance_v6 `
  --per-ip-quota 80 `
  --max-video-seconds 10 `
  --max-video-width 720
```

构建器会输出 `dataset.json`、`assets.jsonl`、`tasks.jsonl` 和跳过批次清单，
同时把任务和媒体索引写入 SQLite。原始素材不会被移动或复制。

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
- IP 的 HMAC 哈希

同一个 `dataset_id + task_id + ip_hash` 只能写入一次。浏览器清除 Cookie
不会绕过这个限制。

每个数据集可以配置 `per_ip_quota`，当前第一批默认允许每个 IP 完成全部 80 道题。

如果你的媒体不放在 `human_review/assets/`，可以配置媒体根目录：

```powershell
$env:HUMAN_REVIEW_ASSET_ROOT = "D:\path\to\your\video-assets"
python human_review/server.py --port 5001
```

也可以指定任务清单和数据库：

```powershell
$env:HUMAN_REVIEW_MANIFEST = "D:\path\to\tasks.jsonl"
$env:HUMAN_REVIEW_DB = "D:\path\to\review.sqlite3"
$env:HUMAN_REVIEW_DATASET = "performance_v6"
python human_review/server.py --port 5001
```

如果服务位于可信反向代理之后，再显式开启代理 IP 读取：

```powershell
$env:HUMAN_REVIEW_TRUST_PROXY_HEADERS = "true"
$env:HUMAN_REVIEW_IP_SECRET = "replace-with-a-stable-secret"
python human_review/server.py --port 5001
```

不要在不可信代理环境中开启 `HUMAN_REVIEW_TRUST_PROXY_HEADERS`。

## 接口

```text
GET  /api/review/next
GET  /api/review/progress
POST /api/review/vote
GET  /api/review/health
GET  /media/asset/<asset-id>
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
