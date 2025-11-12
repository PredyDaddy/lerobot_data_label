# 静止帧五层删除技术方案

## 1. 背景与目标
- 当前 `core/static_frame_processor.py` 将静止帧检测、帧删除与元数据写回耦合在同一类中，仅支持全部静止帧删除且缺乏预览、缓存管理等功能。
- 业务要求在 `core/dataset_manager.py` 中落地完整的静止帧删除服务，实现 **视频帧、Parquet 行、episodes.jsonl、episodes_stats.jsonl、info.json** 五层同步更新。
- 删除流程不再依赖备份机制，需确保在无回滚保障下的安全性（顺序、异常保护、日志）。
- 方案需复用参考仓库 `/home/chenqingyu/robot/new_lerobot` 中的成熟逻辑，特别是元数据更新与统计计算方法，提供可落地的实现细节与与其它模块的适配指南。

- 数据清洗平台的“删除所有静止帧”能力需直接调用本方案新实现的服务接口（见 §8），并以写回原始视频与Parquet为准，确保对“源数据”进行真实变更（非标记、非软删除、非单独缓存覆盖）。
- 服务需提供 REST 与 Python 双接口：REST 供 Web/UI 使用，Python 供内部任务编排；两者均落到同一执行函数，保持一致语义与结果。
- 备份策略：复用现有“删除数据集”时的备份能力——当且仅当当前数据集尚无备份数据集时，首次执行会自动创建备份数据集；若已存在备份数据集则跳过备份创建。执行阶段不启用 dry-run，实际落盘；预览阶段通过检测/片段 API 展示计划即可。

## 2. 数据结构与前置条件
- 数据集目录结构（与当前仓库一致）：
  - `data/chunk-XYZ/episode_XXXXXX.parquet`
  - `videos/chunk-XYZ/observation.images.<camera>/episode_XXXXXX.mp4`
  - `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, `meta/info.json`
- Parquet 文件包含字段：`timestamp`, `frame_index`, `episode_index`, `index`, `action`, `observation.*`, `task_index` 等。
- `info.json` 中包含 `total_frames`, `total_episodes`, `total_videos`, `fps`, `features` 等元数据；`features` 用于识别视频键。
- 所有待删帧索引（同一 episode 内）需在执行前冻结，保证各层删除操作一致；执行过程中不得引入备份。

## 3. 处理流程概览
```
MotionDetectionResult → 帧索引列表
            │
            ├── 视频层删除（全部摄像头，倒序）
            ├── Parquet 数据删除（批量过滤）
            ├── 时间戳&索引重建（timestamp/frame_index/index）
            └── 元数据同步（episodes.jsonl / episodes_stats.jsonl / info.json）
```
- 执行顺序固定为 **视频 → 数据 → 元数据**，防止视频长度变化后索引偏移影响数据层操作。
- 每个阶段需总结成功/失败帧索引，供最终结果返回。

## 4. 帧索引准备
1. 复用现有检测能力：
   - `core/static_frame_processor.py` 中的 `MotionDetector` 与 `VideoMotionAnalyzer` 可抽出到独立模块（命名建议沿用参考仓库 `core/motion_detector.py`），在 `DatasetManager` 初始化时加载配置（阈值、方法、最小静止帧数）。
   - 检测输出 `MotionDetectionResult(frame_index, timestamp, motion_score, is_static)`。服务需提供 `detect_static_frames` 与 `plan_pruning`，对接现有 UI 的预览接口。
2. 生成帧索引列表：
   - 依据策略（全部静止帧、片段采样、智能剪裁），由 `plan_pruning` 输出倒序排序后的帧索引数组。
   - 在执行阶段再次校验：索引必须在 `0 ≤ frame_index < episode_length` 内。

## 5. 视频层删除
### 5.1 路径解析
- 通过 `DatasetManager` 持有的 `DatasetConfig`/`AppConfig` 获取 episode 所在 chunk 目录 (`episode_id // 1000`) 与视频键列表（来自 `info.json['features']` 中 dtype 为 `video` 的键）。
- 支持两种 video_key 输入：完整键（如 `observation.images.laptop`）与短键（如 `laptop`）。应统一映射为目录名。

### 5.2 删除算法（倒序）
- 参考实现：
  ```python
  for video_key in video_keys:
      video_path = dataset/videos/chunk-XYZ/<video_key>/episode_XXXXXX.mp4
      if not video_path.exists():
          record_warning(...)
          continue
      kept_frames = []
      with VideoReader(video_path) as reader:
          for idx, frame in enumerate(reader):
              if idx not in frame_indices_to_drop:
                  kept_frames.append(frame)
      # 写入临时文件，保持原 fps/编码（可用 ffmpeg 通过帧列表重建）
      write_video(temp_path, kept_frames, fps)
      replace(video_path, temp_path)
  ```
- 核心要点：
  - **倒序**迭代 `frame_indices_to_drop` 确保逐帧删除时后续索引不漂移；若使用“重写视频”策略，读取时仍可正序遍历，但需对索引集合使用集合/布尔数组判断。
  - `fps` 使用 `info.json['fps']` 或 `ffprobe` 实际值（参考 `new_lerobot/scripts/data_check/fix_lerobot_timestamps.py` 的 `get_video_metadata` 实现）。
  - 写入时采用临时文件（例如 `episode_XXXXXX.tmp.mp4`），删除成功后原子替换；失败时保留原文件。
  - 对不存在或损坏的视频文件记录 warning，继续处理其它视角。

## 6. Parquet 数据层删除
### 6.1 数据读取与过滤
- 加载 Parquet（使用 `pyarrow` 或 `pandas`）。推荐使用 `pyarrow.parquet.read_table` 减少内存占用。
- 过滤条件：`~df['frame_index'].isin(frame_indices_to_drop)`。
- 删除完成后记录 `removed_count` 与剩余帧数作为后续校验基准。

### 6.2 关键字段重建
- **frame_index**：`np.arange(len(df))`。
- **index**：`np.arange(global_index_start, global_index_start + len(df))`。其中 `global_index_start` 取自原数据第一行的 `index`，确保同 episode 内连续且与其它 episode 无冲突。
- **timestamp**：根据参考项目做法：
  - 若帧率固定，直接 `timestamp = frame_index / fps`。
  - 若原 parquet 存在自定义时间戳，可复用现有列的首值（例如 `start_ts`），按 fps 重建并保留浮点精度；相关逻辑可参考 `new_lerobot/scripts/data_check/fix_lerobot_timestamps.py` 中基于视频时长修正时间戳的方式。
- 将更新后的 DataFrame 写回原路径。

## 7. 元数据同步
### 7.1 `meta/episodes.jsonl`
- 加载 JSONL 后找到 `episode_index == episode_id` 的条目，更新：
  - `length`: 删除后的帧总数。
  - 若存在 `timestamps` 或 `duration` 字段，同步替换为最新值。
- 写回时保持 UTF-8 编码与一行一 JSON 的格式，参考 `new_lerobot/clean_lerobot_dataset.py` 中的 `save_json_lines`。

### 7.2 `meta/episodes_stats.jsonl`
- 基于更新后的 Parquet 重新计算统计。
- 可直接调用参考仓库的 `lerobot/common/datasets/compute_stats.py::compute_episode_stats`：
  1. 构造 `episode_data` 字典，键为 feature 名，值为 numpy 数组或图片路径列表。
  2. 使用 `compute_episode_stats(episode_data, features)` 得到单 episode 统计。
  3. 更新 JSONL 条目中的 `stats` 字段（需序列化为 Python 内建类型，可复用 `serialize_dict` 的逻辑）。

### 7.3 `meta/info.json`
- 更新字段（注意：`total_frames` 为“所有视角视频的总帧数”）：
  - `total_frames`：按“逐 episode × 逐视角”重算，而非简单相减。
    - 设每个 episode 删除后帧长为 `L_e`（来自对应 Parquet 行数），设该 episode 实际存在的视频视角集合为 `V_e`（依据 `features` 中 dtype=video 的键并校验对应视频文件是否存在且非空）。
    - 则 `total_frames = Σ_e ( L_e × |V_e| )`。若某视角的视频在该 episode 缺失或被判定为空文件，则不计入该 episode 的视角数。
  - `total_videos`：重算为所有 episode × 其存在的视频视角数之和；如出现“空视频被删除/替换”则相应递减。
  - `total_episodes`：保持不变（本功能不删除 episode）。
  - `splits.train`：保持 `"0:<total_episodes>"` 格式（可视需要刷新）。
  - `fps`/`features`：若 `features` 中包含视频统计信息（分辨率、编码等），保持不变。
- 写回：使用 `json.dump(..., indent=2, ensure_ascii=False)` 原子替换（先写临时文件再替换），并刷新 `LocalDatasetLoader` 相关缓存（如有）。

## 8. `DatasetManager` 中的新服务设计
### 8.1 类结构建议
```python
class StaticFrameDeletionService:
    def __init__(self, dataset_path: str, config: AppConfig):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.video_fps = self._load_fps()
        self.video_keys = self._load_video_keys()
        self.motion_cache = {}

    def detect_static_frames(...): ...
    def plan_pruning(...): ...
    def delete_frames(episode_id: int, frame_indices: list[int], video_keys: Optional[list[str]] = None): ...
    def _delete_video_frames(...): ...
    def _delete_parquet_rows(...): ...
    def _update_metadata(...): ...
```
- `DatasetManager` 初始化时创建 `self.static_frames = StaticFrameDeletionService(dataset_path, app_config)`，供外部调用。
- `delete_frames` 返回结构：
  ```python
  {
      "success": bool,
      "episode_id": int,
      "requested": N,
      "deleted": M,
      "failed_indices": [...],
      "processing_time": float
  }
  ```

### 8.2 与现有模块的交互
- `core/static_frame_processor.py`：保留检测/计划功能，执行阶段调用 `DatasetManager.static_frames.delete_frames`，逐步淘汰原有的 `LeRobotFrameExtractor` 与 `StaticFrameProcessor.delete_all_static_frames`。
- `web/app.py`：`/api/episodes/<id>/static_prune/execute` 改为：
  ```python
  plan = dataset_manager.static_frames.plan_pruning(...)
  result = dataset_manager.static_frames.delete_frames(episode_id, plan.frame_indices, video_keys=[...])
  ```

### 8.3 对接数据清洗平台 API 设计（前端保持不变）
- REST 接口（路径与协议保持不变）：
  - 检测与预案：
    - GET `/api/episode/<id>/static_frames/<video_key>`（已有）
    - GET `/api/episode/<id>/static_segments/<video_key>`（已有）
  - 执行删除（兼容扩展）：
    - POST `/api/episode/<id>/static_delete`
      - 请求体（兼容现有）：
        ```json
        {
          "video_key": "all"  // 也可为单一视角键，传 "all" 表示对所有视频视角执行统一计划
          // 其他现有字段如 max_deletions/dry_run/create_backup 若存在将被后端忽略或覆盖
        }
        ```
      - 语义：当 `video_key="all"` 时，后端会按配置中的视频键集合自动聚合“静止帧计划”，并对所有视角同步删除；若传入单一 `video_key` 则只针对该视角做计划并进行全视角同步删除（即以该视角的静止帧计划为准）。
      - 聚合策略：默认取“各视角检测结果的交集”（intersection）作为静止帧计划，以保证多视角一致静止；可通过 `config.yaml: static_frame.combine_policy = intersection|union` 配置，union 更激进、intersection 更保守。
      - 备份策略：后端自动复用“删除数据集”的备份能力，若不存在备份数据集则首次执行会创建，若已存在则跳过；无需前端传参。
      - 返回：保持现有响应结构，并可附加 `failed_indices`、`per_view` 等扩展字段（向后兼容）。
- 服务内部路由（实现细节，前端无感）：
  - 后端不再调用 `StaticFrameProcessor.delete_all_static_frames`，而是调用 `DatasetManager.static_frames.plan_pruning(...)` 与 `delete_frames(...)`；检测功能由新文件提供（见 §14）。
  - 执行完成后刷新服务端缓存，前端按原有逻辑刷新该 episode 的视频/表格视图。
- Python 内部调用（供任务编排/脚本使用）：
  - `DatasetManager.static_frames.delete_frames(episode_id: int, frame_indices: list[int], video_keys: Optional[list[str]] = None) -> dict`
  - 与 REST 响应结构保持一致，便于统一日志与指标采集。

- `LocalDatasetLoader`：如内部缓存 episode 列表或 CSV 视图，删除后调用刷新方法（例如重载 parquet）。
- `config.yaml`：新增 `static_frame` 配置块，含 `method`, `threshold`, `min_static_frames`, `combine_policy`, `video_batch_size` 等参数，`AppConfig` 提供 getter。

## 9. 异常处理与日志
- 统一使用 `logging`：
  - `INFO`：记录检测命中率、计划、删除结果。
  - `WARNING`：某个视频缺失 / 单帧失败。
  - `ERROR`：文件读写异常，立即停止后续步骤。

### 9.1 同步处理与一致性校验（事务化步骤）
为保证“真实修改源数据”同时尽可能避免中间态失衡，执行阶段采取严格顺序与幂等控制：
1) 获取 episode 级独占锁（基于文件锁 `.locks/episode_<id>.lock` 或 `fcntl`），拒绝并发写。
2) 生成并冻结预案：`plan = plan_pruning(...)`，得到倒序 `frame_indices` 与目标 `video_keys`；记录预案签名 `plan_hash`（便于幂等）。
3) 视频层重写（逐视角、原子替换）：读取→过滤→写入临时文件→`replace()` 覆盖；失败视角记录 warning 继续，若全部失败则报错退出。
4) Parquet 行删除与关键列重建：过滤→重建 `frame_index/index/timestamp`→写回原路径；失败直接报错终止。
5) 元数据更新：`episodes.jsonl`/`episodes_stats.jsonl`/`info.json` 同步变更；必要时复用 `new_lerobot` 统计函数。
6) 一致性校验：
   - `episodes.jsonl[length] == parquet.rows`
   - `info.json.total_frames` = Σ(episodes.length)
   - 视频时长 ≈ `parquet.rows / fps`（容忍阈值见配置）
7) 记录执行快照（非数据备份）：写入 `meta/ops_logs/static_prune_<episode>_<time>.json`，包含 `plan_hash`、删除计数、失败索引、耗时等，便于追溯。
8) 释放锁；返回统一结果对象（含 `failed_indices`）。

幂等性与重试：
- 若同一 `episode_id + plan_hash` 重复提交，二次执行应快速通过（已删除帧将被视为“已不存在”而不报错）。
- 写入使用临时文件 + 原子替换，避免半写状态；异常退出后不清理原文件。

并发策略：
- 单 episode 写操作串行化（锁）；跨 episode 可并行（视 I/O 资源调整 `video_batch_size`）。
- UI 层对同一 episode 的重复提交禁用提交按钮直至返回或超时。

- 执行逻辑：
  1. 视频层任一视角失败 → 记录失败索引并继续其它视角；最终若所有视角均失败则返回失败。
  2. Parquet 写回失败 → 终止并返回失败；由于无备份，需在文档中提示人工恢复步骤。
  3. 元数据更新失败 → 保留中间状态并返回失败，提示人工使用参考脚本重新计算（见 §10）。

## 10. 参考实现与工具
- `/home/chenqingyu/robot/new_lerobot/clean_lerobot_dataset.py`：整集删除时的元数据更新流程，可复用 `load_json_lines` / `save_json_lines`、重新编号逻辑。
- `/home/chenqingyu/robot/new_lerobot/lerobot/common/datasets/compute_stats.py`：`compute_episode_stats`、`aggregate_stats`、统计值结构定义。
- `/home/chenqingyu/robot/new_lerobot/scripts/data_check/fix_lerobot_timestamps.py`：基于视频元数据更新 `timestamp` 的范例，可借鉴 `ffprobe` 的调用方式。
- `/home/chenqingyu/robot/new_lerobot/scripts/data_convert/...`：构造 `frame_index`/`timestamp`/`index` 列的规范（例如 `frame_index = torch.arange(0, num_frames)`）。

## 11. 测试与验证策略
1. **单元测试**（新增 `tests/test_static_frame_deletion.py`）：
   - 模拟小型数据集（使用 `tmp_path`）执行删除，断言五层数据保持一致。
   - 验证帧索引倒序处理正确（删除多个连续/非连续索引）。
   - 测试无视频文件、部分索引缺失等异常分支。
2. **集成测试**：
   - 使用真实数据副本（例如 `datasets/grasp_dataset` 的子集）执行一次静止帧删除，通过 UI 验证视频播放顺序与 CSV 视图无断帧。
   - 调用 `DatasetLoader` 加载同一 episode，确认 `frame_index` 与 `timestamp` 连续且单调递增。
3. **性能评估**：
   - 记录视频重编码与 Parquet 重写耗时，必要时考虑批量处理或异步化。

## 12. 与现有功能的适配总结
- `core/static_frame_processor.py`：移除该文件；静止帧检测/分析能力重构至新文件 `core/motion_detector.py`（见 §14），仅保留检测与计划功能。
- `core/dataset_manager.py`：实现静止帧删除服务 `StaticFrameDeletionService`，对外提供 `detect`/`plan`/`delete` 方法，并在 `DatasetManager` 初始化时注入。
- `web/app.py` 与前端 JS：端点与 UI 交互保持不变；服务端内部从旧实现切换为 `DatasetManager.static_frames`，兼容新增 `video_key="all"` 的扩展值（可不改前端）。
- 文档更新：本方案文件加入 `docs/`，并在 README 或 `docs/xxx` 中添加链接，让团队了解新流程。
- 数据清洗平台 UI 对接步骤（保持不变的同时，增强后端能力）：
  1) 继续使用现有的“一键检测静止帧”与“一键删除”按钮，分别调用现有检测 API 与 POST `/api/episode/<id>/static_delete`。
  2) 如需全视角删除，可将 `video_key` 传 `"all"`（后端自动聚合所有视角的静止帧计划并同步删除）。
  3) 收到响应后根据 `failed_indices` 做提示；若 `success=false`，展示错误并提供“查看执行快照”的入口（读取 `meta/ops_logs`）。
  4) 前端在执行期间禁用按钮并显示进度；完成后刷新该 episode 的视频与表格视图。
- 真实数据变更告警：在 UI 二次确认弹窗中明确提示“将直接修改原始视频与Parquet数据，不可自动回滚”，并提供参考脚本链接（见 §10）。

- 配置：`config.yaml` 新增静止帧参数块，`AppConfig` 提供读取，避免硬编码阈值。

## 13. 风险与后续工作
- 无备份模式下的失败恢复需依赖手工操作：建议保留 `new_lerobot` 中的 `clean_lerobot_dataset.py` 及统计脚本作为应急工具。
- 视频重编码耗时较长，可评估批量帧丢弃算法（例如 ffmpeg 的 `select='not(in(n,idx1,idx2,...))'`）。
- 若未来需要批量 episode/帧删除，可在本服务基础上封装更高层的任务队列与进度汇报。

---
本文档用于指导在 `dataset_management.py` 中实现静止帧五层删除服务，供研发与测试团队协作参考。


## 14. 文件重构与兼容策略
- 文件调整：
  - 移除 `core/static_frame_processor.py`（原实现包含检测+删除+写回三合一，现拆分）。
  - 新增 `core/motion_detector.py`：仅承载“静止帧检测/分析/片段规划”能力，包含：
    - `MotionDetectionConfig`、`MotionDetectionResult`
    - `MotionDetector`（预处理+帧差/可扩展算法）
    - `VideoMotionAnalyzer`（结果聚合为片段、合并/交并策略）
  - 删除执行与五层写回由 `core/dataset_manager.py` 的 `StaticFrameDeletionService` 负责。
- 前端保持不变：现有检测端点与删除端点路径/参数不变；服务端内部从旧处理器切换为“检测模块 + DatasetManager 删除服务”的组合实现。
- 一键检测/一键删除：
  - “一键检测所有静止帧”沿用现有检测端点；当请求聚合全视角时，后端在 `plan_pruning` 内部串行/并行调用各视角检测并依据 `combine_policy` 生成统一帧索引。
  - “一键删除”沿用 POST `/api/episode/<id>/static_delete`；当 `video_key="all"` 时按统一帧索引对全部视角同步删除并写回五层数据。
- 备份复用：删除前调用“删除数据集”已有的备份创建能力（仅当不存在备份数据集时创建），以保证首次操作具备回退手段；后续重复执行自动跳过备份创建。
- 写回与缓存刷新：所有文件写回均采用“临时文件 + 原子替换”；执行完毕后刷新 `LocalDatasetLoader` 等读缓存，前端按既有逻辑刷新页面即可看到最新状态。
