# LeRobot 数据集可视化与标注平台

一个功能完整的LeRobot数据集可视化和标注平台，支持数据浏览、智能标注、静止帧检测和数据集管理等功能。

## 🌟 主要特性

### 📊 数据集可视化
- **多模态数据展示** - 支持时序数据图表和多视角视频同步播放
- **交互式图表** - 基于Dygraphs的实时数据缩放和平移
- **Episode浏览** - 快速切换不同episode进行对比分析
- **数据统计** - 实时显示数据集基本信息和统计指标

### 🏷️ 智能标注系统
- **分段标注** - 支持按时间轴对episode进行技能分段标注
- **技能管理** - 预定义技能类型和自定义技能标签
- **标注历史** - 自动保存标注历史，支持撤销和重做
- **数据验证** - 自动校验标注数据的完整性和一致性

### 🎬 静止帧检测与优化
- **智能检测** - 基于帧差法的静止帧自动识别
- **多视角融合** - 支持多个摄像头视角的检测结果合并
- **安全删除** - 五层数据同步删除机制，确保数据一致性
- **批量处理** - 支持episode级别的静止帧批量清理

### 📁 数据集管理
- **Episode管理** - 安全的episode删除和重新编号
- **数据备份** - 自动备份机制，支持数据恢复
- **技能分割** - 基于标注技能的智能数据集分割
- **格式转换** - 支持LeRobot标准格式的导入导出

## 🛠️ 环境配置

### 系统要求
- Python 3.11+
- FFmpeg（用于视频处理）
- 足够的磁盘空间用于数据集和备份

### 1. 创建Conda环境

```bash
# 创建名为lerobot_data的conda环境
conda create -n lerobot_data python=3.11 -y

# 激活环境
conda activate lerobot_data
```

### 2. 安装依赖包

```bash
# 安装基础依赖
pip install -r requirements.txt

# 手动安装PIL（如果未包含在requirements.txt中）
pip install Pillow
```

### 3. 验证安装

```bash
# 检查关键包
python -c "import flask, pandas, numpy, cv2, PIL; print('所有依赖包安装成功！')"
```

### 4. FFmpeg安装（可选但推荐）

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并安装
```

## 🚀 快速开始

### 1. 准备数据集

确保你有一个LeRobotv2.1格式的数据集，目录结构应如下：

```
your_dataset/
├── meta/
│   ├── info.json          # 数据集基本信息
│   ├── episodes.jsonl     # episode元数据
│   └── episodes_stats.jsonl  # 统计信息
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet  # 时序数据
└── videos/
    └── chunk-000/
        ├── observation.images.laptop/
        │   └── episode_000000.mp4   # 视频数据
        └── observation.images.phone/
            └── episode_000000.mp4
```

### 2. 配置数据集路径

编辑 `config.yaml` 文件，修改数据集路径：

```yaml
dataset:
  path: "/path/to/your/dataset"  # 修改为你的数据集路径
```

### 3. 启动应用

```bash
# 使用默认配置启动
python main.py

# 或指定数据集路径和端口
python main.py --dataset-path /path/to/your/dataset --port 9090
```

### 4. 访问Web界面

打开浏览器访问：`http://127.0.0.1:9090`

主要功能页面：
- **主页** - 数据集选择和概览
- **可视化页面** - `/episode_0` 等查看具体episode
- **标注页面** - `/annotate/0` 等进行数据标注

### 5. 示例命令

```bash
# 启动服务器（指定数据集和端口）
python main.py --dataset-path /mnt/data/kouyouyi/dataset/bottle_handover --port 9090

# 启用调试模式
python main.py --debug --verbose

# 指定配置文件
python main.py --config my_config.yaml
```

## 📂 代码架构

```
lerobot_data/
├── main.py                 # 应用入口点和命令行处理
├── config.yaml            # 配置文件
├── requirements.txt       # Python依赖
├── core/                  # 核心业务逻辑
│   ├── config.py         # 配置管理类
│   ├── data_loader.py    # 数据集加载器
│   ├── video_handler.py  # 视频处理模块
│   ├── annotation_store.py  # 标注数据存储
│   ├── dataset_manager.py   # 数据集管理器
│   ├── motion_detector.py   # 静止帧检测
│   └── split_by_skill.py    # 技能分割模块
├── web/                   # Web应用层
│   ├── app.py            # Flask应用和API路由
│   └── __init__.py       # Web模块初始化
├── templates/            # HTML模板
│   ├── home.html         # 主页模板
│   ├── visualize_dataset_template.html  # 可视化页面
│   └── annotate.html     # 标注页面模板
├── static/               # 静态资源
│   ├── css/             # 样式文件
│   ├── js/              # JavaScript文件
│   └── images/          # 图片资源
└── scripts/             # 工具脚本，开发时使用
    ├── count_video_frames.py # 计算视频帧数
    └── analyze_parquet.py # 分析parquet文件结构
```

### 核心模块说明

#### `core/config.py`
- **AppConfig** - 应用配置管理，支持YAML配置文件和默认值
- **DatasetConfig** - 数据集路径模板和配置管理

#### `core/data_loader.py`
- **LocalDatasetLoader** - LeRobot格式数据集的统一加载器
- **IterableNamespace** - 支持点号访问的动态配置对象
- 负责Parquet数据读取和CSV格式转换

#### `core/video_handler.py`
- **VideoHandler** - 视频文件处理和路径管理
- 集成静止帧检测功能
- 支持多视角视频的同步处理

#### `core/annotation_store.py`
- **AnnotationStore** - 标注数据的持久化存储
- 支持JSON和JSONL格式的读写
- 提供数据校验和原子写入机制

#### `core/dataset_manager.py`
- **DatasetManager** - 数据集的高级管理功能
- **StaticFrameDeletionService** - 静止帧删除服务
- 支持episode删除、备份创建、技能分割等

#### `core/motion_detector.py`
- **MotionDetector** - 静止帧检测算法实现
- **VideoMotionAnalyzer** - 视频级别的运动分析
- 支持多种检测方法和参数配置

#### `web/app.py`
- Flask Web应用的主入口
- 定义所有API端点和Web路由
- 集成所有核心模块提供Web服务

## ⚙️ 配置说明

### config.yaml 完整配置项

```yaml
# 应用配置
app:
  host: "127.0.0.1"        # 服务器绑定地址
  port: 9090               # 服务器端口
  debug: false             # 调试模式

# 数据集配置
dataset:
  path: "/path/to/dataset" # 数据集路径
  cache_enabled: false     # 数据缓存开关

# UI配置
ui:
  episodes_per_page: 50    # 每页显示episode数量
  default_video_keys:      # 默认显示的视频视角
    - "laptop"
    - "phone"
  chart_height: 400        # 图表高度(像素)
  video_width: 640         # 视频播放器宽度

# 标注配置
ui:
  annotation:
    autosave_on_next: true      # 切换下一个episode时自动保存
    write_mode: "append"        # 写入模式
    output_format: "jsonl"      # 输出格式
    default_skills:             # 默认技能类型
      - "Pick"
      - "Place"
      - "Move"
      - "Rotate"
      - "Push"
      - "Pull"
      - "Grasp"
      - "Release"
    persist_dataset_in_session: true  # 会话中保持数据集选择

# 静止帧检测配置
motion_detection:
  enabled: true              # 是否启用静止帧检测
  method: "frame_diff"       # 检测方法
  threshold: 0.02            # 静止阈值
  min_static_frames: 5       # 最小连续静止帧数
  resize_width: 320          # 处理时图像宽度
  resize_height: 240         # 处理时图像高度
  gaussian_blur_kernel: 5    # 高斯模糊核大小
  cache_results: true        # 是否缓存检测结果
  cache_duration_hours: 24   # 缓存有效期(小时)
  combine_policy: "intersection"  # 多视角合并策略
  video_batch_size: 4        # 视频批处理大小

# 日志配置
logging:
  level: "INFO"              # 日志级别
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 性能配置
performance:
  max_csv_size: 10485760     # 最大CSV文件大小 (10MB)
  request_timeout: 30        # 请求超时时间(秒)
  static_cache_timeout: 3600 # 静态文件缓存时间(秒)
```

## 📝 功能特性详解

### 数据集可视化
- **多视角同步播放** - 支持多个摄像头视角的时间同步视频播放
- **实时数据图表** - 基于Dygraphs的交互式时序数据图表
- **智能数据采样** - 大数据集的自动降采样和性能优化
- **Episode导航** - 快速切换不同episode进行对比分析

### 智能标注系统
- **时间轴标注** - 直观的拖拽式时间轴分段标注
- **技能标签管理** - 预定义技能类型和自定义标签支持
- **标注历史记录** - 完整的标注操作历史和版本管理
- **数据完整性校验** - 自动检测和防止标注数据冲突

### 静止帧检测与删除
- **帧差法检测** - 基于像素差异的静止帧识别算法
- **多视角融合** - 支持交集和并集两种多视角合并策略
- **五层同步删除** - 视频、数据、元数据的完整同步删除
- **安全备份机制** - 删除前自动备份，支持数据恢复

### 数据集管理
- **Episode级别管理** - 安全的episode删除和智能重新编号
- **智能数据分割** - 基于技能标注的自动化数据集分割
- **格式兼容性** - 完全兼容LeRobot标准数据格式
- **批量操作支持** - 支持大规模数据集的批量处理

## 🔌 API文档

### 数据集相关API

#### 获取数据集信息
```http
GET /api/dataset/info
```

#### 获取Episode列表
```http
GET /api/episodes?page=1&per_page=50
```

#### 获取Episode数据
```http
GET /api/episode/{episode_id}/data
```

#### 获取Episode视频信息
```http
GET /api/episode/{episode_id}/videos
```

### 标注相关API

#### 获取Episode标注
```http
GET /api/annotations/{episode_id}
```

#### 保存Episode标注
```http
POST /api/annotations/{episode_id}
Content-Type: application/json

{
  "episode_id": 0,
  "task_name": "抓取物体",
  "init_scene_text": "桌面有一个杯子",
  "label_info": {
    "action_config": [
      {
        "start_frame": 0,
        "end_frame": 100,
        "skill": "Pick",
        "action_text": "接近并抓取杯子"
      }
    ]
  }
}
```

#### 获取标注统计
```http
GET /api/annotations/stats
```

### 静止帧检测API

#### 检测静止帧
```http
GET /api/episode/{episode_id}/static_frames/{video_key}?threshold=0.02&min_static_frames=5
```

#### 获取静止片段
```http
GET /api/episode/{episode_id}/static_segments/{video_key}
```

#### 删除静止帧
```http
POST /api/episode/{episode_id}/delete_frames
Content-Type: application/json

{
  "deletion_type": "static_frames",
  "video_key": "laptop",
  "create_backup": true
}
```

### 数据集管理API

#### 删除Episode
```http
DELETE /api/episode/{episode_id}?backup=true
```

#### 创建备份
```http
POST /api/dataset/backup
```

#### 按技能分割数据集
```http
POST /api/dataset/split_by_skill
Content-Type: application/json

{
  "output_root": "/path/to/output/directory"
}
```

## ❓ 常见问题

### Q: 启动时提示"数据集路径不存在"
A: 检查以下几点：
1. 确保`config.yaml`中的数据集路径正确
2. 数据集目录存在且包含`meta/info.json`文件
3. 使用命令行参数`--dataset-path`指定正确路径

### Q: 视频无法播放
A: 可能的原因：
1. FFmpeg未安装或版本不兼容
2. 视频文件格式不支持（仅支持MP4、WebM、AVI）
3. 视频文件损坏或路径不正确

### Q: 静止帧检测结果不准确
A: 可以尝试：
1. 调整`motion_detection.threshold`参数（降低阈值检测更多静止帧）
2. 调整`min_static_frames`参数（增加最小连续帧数）
3. 检查视频质量和光照条件

### Q: 标注数据保存失败
A: 检查以下几点：
1. 确保数据集目录有写入权限
2. 检查标注数据格式是否正确
3. 查看服务器日志获取详细错误信息

### Q: Web界面加载缓慢
A: 优化建议：
1. 减少每页显示的episode数量
2. 启用数据缓存功能
3. 确保视频文件不是过大
4. 检查网络连接和服务器性能

### Q: 如何处理大数据集
A: 建议：
1. 使用数据分页功能
2. 启用数据采样和缓存
3. 考虑使用SSD存储
4. 增加服务器内存
