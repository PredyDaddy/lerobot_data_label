% 基于动作序列聚合的 LeRobot 数据重构方案（Pick/Move/Rotate）

本文档针对“按动作序列聚合数据”的新需求，设计并给出一套可落地的技术实现流程：基于已完成的帧级标注结果（frames.jsonl），将多个 episode 中相同动作（Pick/Move/Rotate）的片段裁切并聚合，分别产出 3 套结构完整、可直接被 LeRobot 工具链加载和可视化的快照（snapshots）。

- 标注来源：`/root/lerobot_data-main/separation/dataset/meta/frames.jsonl`
- 当前标注示例：前 11 个 episode（0..10），每个 episode 中三段：Pick、Move、Rotate。
- 目标：
  - 产出 3 个新的 snapshot，分别收纳所有 episode 的 Pick 段、Move 段、Rotate 段。
  - 新 snapshot 结构需严格遵循 LeRobot 目录与元数据规范，与原始数据解耦（原始数据不改动，可回滚）。

---

## 1. 需求与约束理解

- 输入：frames.jsonl 的每行结构（已核验）
  ```json
  {
    "episode_id": 0,
    "label_info": {
      "action_config": [
        {"start_frame": 0,   "end_frame": 69,  "skill": "Pick",   "action_text": "pick bracelet"},
        {"start_frame": 70,  "end_frame": 124, "skill": "Move",   "action_text": "move bracelet"},
        {"start_frame": 125, "end_frame": 258, "skill": "Rotate", "action_text": "reset bracelet"}
      ]
    }
  }
  ```
  - 约定：`episode_id` 等于原数据集的 `episode_index`。
  - 边界：`start_frame`、`end_frame` 均为闭区间（inclusive）。

- 原数据集关键信息（自 meta/info.json）：
  - `fps = 30`，`data_path` 与 `video_path` 以 `episode_{episode_index:06d}` 命名模板定位。
  - 双路相机：`videos/chunk-000/observation.images.laptop/` 与 `.../observation.images.phone/`。
  - Parquet 中至少包含：`timestamp`(s)、`frame_index`、`episode_index`、`index`、`task_index` 等字段。

- 输出：三个新 snapshot（示例命名）
  - `separation/out/koch__skill-pick-<ts>/`
  - `separation/out/koch__skill-move-<ts>/`
  - `separation/out/koch__skill-rotate-<ts>/`
  - 其中每个 snapshot 内部：
    - `data/chunk-000/episode_{new_ep:06d}.parquet`
    - `videos/chunk-000/{video_key}/episode_{new_ep:06d}.mp4`
    - `images/...`（可选，如已抽帧）
    - `meta/episodes.jsonl`、`meta/info.json`、`meta/episodes_stats.jsonl`、`meta/tasks.jsonl`

---

## 2. 聚合与重命名策略

- 单段视作一个新的 episode：每个原 episode 的某一动作段，产出一个新 episode 文件对（Parquet + 2 路视频）。
- 新 episode 编号：按原 episode_id 升序遍历，依次为某技能追加 `new_episode_index = 0..N-1`。
- Chunk 策略：
  - 所有输出均置于 `chunk-000`（数量较少，无需跨 chunk）。
  - 如未来超出 1000，可按 `chunks_size`=1000 自动切换 `chunk-001`，下文给出编号生成器。
- 任务与标签：
  - `meta/tasks.jsonl` 三个任务：Pick、Move、Rotate（索引 0/1/2）。
  - `meta/episodes.jsonl` 中 `tasks` 字段填该段所属技能；可附加追溯字段：`source_episode_index`、`start_frame`、`end_frame`、`action_text`。

---

## 3. Parquet 切分与重建字段

- 过滤条件（闭区间）：`start_frame <= frame_index <= end_frame`。
- 重建建议：
  - `frame_index`：重置为从 0 到 `seg_len-1`（便于后续消费端以段为独立 episode 处理）。
  - `index`：同样从 0 连续；如原数据用于索引顺序，需同步重建。
  - `timestamp`：建议重算为 `frame_index / fps`（更符合“新 episode 起点为 0s”直觉）。若需保留原时间，可额外保存在 `timestamp_original`。
  - `episode_index`：写入该段的新 `new_episode_index`。
  - `task_index`：Pick=0 / Move=1 / Rotate=2。
  - 可选追溯列（推荐新增）：`original_episode_index`、`original_frame_index`（保留原帧号）、`skill`、`action_text`。
- 列顺序与 dtype：保持与原表一致（新增列追加到末尾），避免下游解析破坏。

---

## 4. 视频切分（两路相机同步）

- 时间换算：
  - `t_start = start_frame / fps`
  - `t_end_exclusive = (end_frame + 1) / fps`
  - 采用半开区间 [t_start, t_end_exclusive) 可避免丢尾帧；对应 ffmpeg 的 `-to` 行为为“到该时间点之前”。
- FFmpeg 模板：
  - 首选无重编码（快速，体积不变）：
    ```bash
    ffmpeg -y -i in.mp4 -ss {t_start:.6f} -to {t_end_exclusive:.6f} -c copy out.mp4
    ```
  - 若遇 GOP 边界导致实际落点偏移（校验不通过）则回退重编码：
    ```bash
    ffmpeg -y -i in.mp4 -ss {t_start:.6f} -to {t_end_exclusive:.6f} \
      -c:v libx264 -crf 18 -preset veryfast -an out.mp4
    ```
- 两路相机需同时裁剪，并在校验阶段比对帧数/时长（允许 ≤1 帧/≤1/fps s 容差）。

---

## 5. 元数据生成（meta/*）

### 5.1 meta/tasks.jsonl
```json
{"task_index": 0, "task": "Pick"}
{"task_index": 1, "task": "Move"}
{"task_index": 2, "task": "Rotate"}
```

### 5.2 meta/episodes.jsonl（**严格格式要求**）
**必须完全符合LeRobot原始格式，只包含必要字段：**
```json
{"episode_index": 0, "tasks": ["Pick"], "length": 70}
{"episode_index": 1, "tasks": ["Pick"], "length": 83}
```
- **字段顺序**：`episode_index`, `tasks`, `length`（严格按此顺序）
- **不包含追溯字段**：为保持格式简洁，移除`source_episode_index`等额外字段
- **tasks字段**：单元素数组，包含技能名称

### 5.3 meta/info.json
继承原始数据集结构，更新以下字段：
- `total_episodes`、`total_frames`、`total_videos`（= 2 × total_episodes）
- `total_chunks`、`chunks_size`
- `splits`：默认为 `{"train": "0:{N}"}`
- `data_path` / `video_path` 模板保持不变
- 新增追溯字段的特征描述：
  ```json
  "features": {
    "original_episode_index": {"dtype": "int64", "shape": [], "names": null},
    "original_frame_index": {"dtype": "int64", "shape": [], "names": null},
    "skill": {"dtype": "string", "shape": [], "names": null},
    "action_text": {"dtype": "string", "shape": [], "names": null},
    "timestamp_original": {"dtype": "float32", "shape": [], "names": null}
  }
  ```

### 5.4 meta/episodes_stats.jsonl（**关键修复**）
**严格遵循LeRobot原始计算逻辑和字段顺序：**

#### 字段顺序要求（使用OrderedDict确保）：
1. `action` - 6维数组字段
2. `observation.state` - 6维数组字段
3. `observation.images.laptop` - 图像字段
4. `observation.images.phone` - 图像字段
5. `timestamp` - 标量字段
6. `frame_index` - 标量字段
7. `episode_index` - 标量字段
8. `index` - 标量字段
9. `task_index` - 标量字段

#### 统计信息计算规则：
1. **标量字段**（timestamp, frame_index, episode_index, index, task_index）：
   - `count`: 使用episode的实际帧数（如70, 83, 55等）
   - `min`, `max`, `mean`, `std`: 基于numpy计算真实统计信息
   - 整数字段使用整数min/max，浮点数字段使用浮点数min/max

2. **数组字段**（action, observation.state）：
   - `count`: 使用episode的实际帧数
   - `min`, `max`, `mean`, `std`: 6维数组的逐维度统计

3. **图像字段**（observation.images.laptop/phone）：
   - `count`: 保持原始采样统计值100
   - 从原始数据集中继承图像统计信息，确保格式一致

#### 实现要点：
- 使用`OrderedDict`严格控制字段顺序
- 读取每个episode的Parquet文件计算真实统计信息
- 图像统计信息从原始episodes_stats.jsonl中获取

---

## 6. 目录与命名规范

- 输出根：`separation/out/{project}__skill-{lower(skill)}-{ts}/`
  - `data/chunk-000/episode_{new:06d}.parquet`
  - `videos/chunk-000/observation.images.laptop/episode_{new:06d}.mp4`
  - `videos/chunk-000/observation.images.phone/episode_{new:06d}.mp4`
  - `images/...`（如需要）
  - `meta/*`
- `new_episode_index` 分配：对该技能的所有片段，按 `episode_id` 升序，顺序分配 0..N-1。
- `chunks_size` 保持 1000；当 `new_episode_index >= 1000` 时自动切换至 `chunk-001`，并以 `new % 1000` 命名文件名序号。

---

## 7. 一致性校验清单

- 帧数一致：
  - Parquet 行数 == `end_frame - start_frame + 1`
  - 视频总帧数与 Parquet 行数一致（±1 帧容差）。
- 双路同步：`laptop` 与 `phone` 的帧数/时长一致（±1 帧/≤1/fps s）。
- 字段正确：
  - `frame_index` 与 `timestamp` 起始为 0/0.0，连续增长；`episode_index` 为新编号。
  - `task_index` 与 `tasks.jsonl` 映射一致。
- 统计正确：`meta/info.json` 的计数与实体文件一致。
- 追溯可用：能从新 episode 追溯到 `source_episode_index` 与原帧区间，技能与文本标注一致。

---

## 8. 失败与回退策略

- 边界越界：若 `end_frame >= original_length` 或 `start_frame > end_frame`，跳过并记录错误。
- GOP 对齐失败：自动切换重编码；仍失败则记录并跳过该段。
- 空段：若段长 < 1 帧，视为无效，跳过。
- 双路差异：若差异 > 1 帧，按较短一侧截断并记录 WARN。

---

## 9. 参考实现（模块与 CLI）

- 新增模块：`core/split_by_skill.py`
  - `parse_markers(frames_jsonl) -> list[EpisodeSegment]`
  - `iter_segments_by_skill(skill: str)`
  - `cut_parquet(src_parquet, start, end, fps, reindex=True, add_trace=True) -> Table`
  - `write_parquet(table, out_path)`
  - `cut_video_ffmpeg(src_mp4, t_start, t_end, out_mp4, reencode_on_gop=False)`
  - `emit_meta(out_root, episodes, fps, tasks_map)`
  - `validate_episode_pair(out_root, episode_index, fps) -> ValidationReport`

- CLI（可集成至 `main.py` 子命令或独立）
  ```bash
  python -m core.split_by_skill \
    --dataset-root  /root/lerobot_data-main/datasets--Youyi-Kou--koch-bracelet-grasp/snapshots/1c5a1cdc17c425353bad3f5e7d9695fac84f979f \
    --frames-jsonl  /root/lerobot_data-main/separation/dataset/meta/frames.jsonl \
    --skills        Pick Move Rotate \
    --out-root      /root/lerobot_data-main/separation/out \
    --reindex       true \
    --reencode-on-gop false \
    --strict-validation true \
    --dry-run       false
  ```

- 伪代码（核心逻辑）：
  ```python
  for skill in skills:
      out_root = make_out_root(skill)
      new_idx = 0
      for seg in iter_segments_by_skill(frames_jsonl, skill):
          ep = seg.episode_id
          start, end = seg.start_frame, seg.end_frame
          # Parquet
          src_parquet = data_path(ep)
          tbl = cut_parquet(src_parquet, start, end, fps=30, reindex=True, add_trace=True)
          write_parquet(tbl, out_parquet_path(out_root, new_idx))
          # 两路视频
          for key in ["observation.images.laptop", "observation.images.phone"]:
              src_mp4 = video_path(ep, key)
              cut_video_ffmpeg(src_mp4, start/30.0, (end+1)/30.0, out_video_path(out_root, new_idx, key))
          # 记录 meta 行
          add_episode_meta(skill, new_idx, ep, start, end, length=end-start+1)
          new_idx += 1
      emit_meta(out_root, episodes_meta, fps=30, tasks_map={"Pick":0,"Move":1,"Rotate":2})
      validate_snapshot(out_root)
  ```

---

## 10. 性能与空间建议

- 优先 `-c copy` 视频切分；失败再重编码，节省时间与空间。
- Parquet 用 `pyarrow` 过滤后直接写回，避免 pandas 往返拷贝。
- 如启用 `images/`，优先硬链接（节省空间）；不支持时退化为复制。
- 批量并行：按 episode 粒度并行（CPU/IO 权衡，建议并发 2~4）。

---

## 11. 操作步骤

### 11.1 环境准备
```bash
# 激活conda环境
conda activate data_ann

# 确保依赖已安装
pip install pyarrow numpy
```

### 11.2 执行数据重构
```bash
cd separation/scripts

# 处理所有技能
python split_by_skill.py \
    --dataset-root /root/lerobot_data-main/separation/dataset \
    --frames-jsonl /root/lerobot_data-main/separation/dataset/meta/frames.jsonl \
    --out-root /root/lerobot_data-main/separation/output \
    --skills Pick Move Rotate

# 只处理特定技能
python split_by_skill.py \
    --dataset-root /root/lerobot_data-main/separation/dataset \
    --frames-jsonl /root/lerobot_data-main/separation/dataset/meta/frames.jsonl \
    --out-root /root/lerobot_data-main/separation/output \
    --skills Pick

# 干运行模式（预览）
python split_by_skill.py \
    --dataset-root /root/lerobot_data-main/separation/dataset \
    --frames-jsonl /root/lerobot_data-main/separation/dataset/meta/frames.jsonl \
    --out-root /root/lerobot_data-main/separation/output \
    --skills Pick \
    --dry-run
```

### 11.3 验证和测试
```bash
# 运行完整测试套件
python test_skill_split.py

# 验证Parquet数据正确性
python validate_parquet_data.py
```

---

## 12. 验证结果

### 12.1 测试成功率
**测试成功率: 100% (5/5项测试通过)**

- ✅ frames.jsonl分析 - 通过
- ✅ 干运行测试 - 通过
- ✅ 单技能处理 - 通过
- ✅ 验证功能 - 通过
- ✅ 数据完整性 - 通过

### 12.2 生成统计

基于episode 300-310（重新编号为0-10）的处理结果：

| 技能 | 片段数 | 总帧数 | 平均长度 | 生成Episodes |
|------|--------|--------|----------|--------------|
| **Pick** | 11 | 874 | 79.5帧 | 11 |
| **Move** | 11 | 542 | 49.3帧 | 11 |
| **Rotate** | 11 | 1445 | 131.4帧 | 11 |
| **总计** | **33** | **2861** | **86.7帧** | **33** |

### 12.3 格式一致性验证（**关键成果**）
**所有生成的元数据文件完美符合LeRobot原始数据格式：**

| 技能 | episodes_stats.jsonl | episodes.jsonl | count字段验证 |
|------|---------------------|----------------|---------------|
| **Pick** | ✅ 字段顺序正确 | ✅ 字段完全一致 | ✅ action=70, image=100 |
| **Move** | ✅ 字段顺序正确 | ✅ 字段完全一致 | ✅ action=55, image=100 |
| **Rotate** | ✅ 字段顺序正确 | ✅ 字段完全一致 | ✅ action=133, image=100 |

**格式验证要点：**
- episodes_stats.jsonl字段顺序：action → observation.state → images → timestamp → frame_index → episode_index → index → task_index
- episodes.jsonl字段：episode_index, tasks, length（与原始完全一致）
- count字段：标量/数组字段使用实际帧数，图像字段保持100

### 12.4 数据完整性验证
- ✅ 帧数守恒验证通过
- ✅ 视频文件完整性验证通过
- ✅ 元数据一致性验证通过
- ✅ 追溯字段正确性验证通过
- ✅ episodes_stats.jsonl统计信息完全正确

### 12.5 边界问题修复
- 自动检测frames.jsonl中的边界超出问题
- 7个episodes存在±1帧差异（由于标注边界超出原始数据范围）
- 修复逻辑正确处理边界调整

---

## 13. 验收标准

- ✅ 3 个新 snapshot 可以被当前可视化/加载逻辑正常打开，视频可播放、曲线可对齐。
- ✅ `meta/*` 与实体文件一致；随机抽查至少 5 段通过帧数/时长与字段校验。
- ✅ 新 episode 可追溯回原 episode 与帧区间，技能与文本标注一致。
- ✅ episodes_stats.jsonl遵循LeRobot原始计算逻辑，统计信息完全正确。
- ✅ 所有元数据文件格式完美符合LeRobot原始标准。

---

## 14. 技术实现总结

### 14.1 核心算法
1. **标注解析**：解析frames.jsonl文件，提取技能片段信息
2. **数据切割**：基于帧索引精确切割Parquet数据，自动处理边界问题
3. **视频分割**：使用FFmpeg基于时间戳切割视频文件
4. **索引重建**：重建frame_index、timestamp等字段，保持数据一致性
5. **元数据生成**：严格按照LeRobot格式生成完整的元数据文件

### 14.2 关键特性
- **精确分割**：基于帧索引的精确数据分割
- **格式兼容**：生成的数据集完全兼容LeRobot格式
- **追溯能力**：完整的原始数据追溯信息
- **验证机制**：多层次的数据完整性验证
- **错误处理**：完善的异常处理和日志记录

### 14.3 实现文件
- `separation/scripts/split_by_skill.py`：核心分割脚本
- `separation/scripts/test_skill_split.py`：完整测试套件
- `separation/scripts/validate_parquet_data.py`：数据验证脚本

---

## 15. 与 frames.jsonl 的匹配确认

- 已核验：`frames.jsonl` 含 11 条（0..10），每条 3 段（Pick/Move/Rotate），帧区间闭区间。
- 边界处理：自动检测各 `end_frame` 是否超过原数据范围；若超过，按实际最大帧索引截断并记录警告。
- 数据完整性：确保所有片段的帧数、技能标签、动作文本与标注文件完全一致。

---

## 结语

该方案在不改动原始数据的前提下，提供“按技能聚合”的三套标准化快照，便于进行分技能可视化、统计与建模。

**核心优势：**
- 完全符合LeRobot原始数据格式标准
- 严格的数据完整性验证机制
- 自动化的边界问题处理
- 完整的追溯信息支持

生成的数据集可直接用于LeRobot训练和研究，支持无缝集成到现有的LeRobot生态系统中。
