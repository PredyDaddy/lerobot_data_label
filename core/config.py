"""
配置管理模块
功能: 路径模板、应用配置、常量定义
"""

from pathlib import Path
import yaml
from typing import Dict, Any


class DatasetConfig:
    """数据集配置管理"""

    # 路径模板 (来自info.json)
    DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    # 元数据路径
    INFO_PATH = "meta/info.json"
    EPISODES_PATH = "meta/episodes.jsonl"
    STATS_PATH = "meta/episodes_stats.jsonl"
    FRAMES_PATH = "meta/frames.jsonl"

    # 默认设置
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_FPS = 30

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def get_data_path(self, episode_index: int, chunk_size: int = None) -> Path:
        """获取数据文件路径"""
        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        episode_chunk = episode_index // chunk_size
        return self.dataset_path / self.DATA_PATH_TEMPLATE.format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )

    def get_video_path(self, episode_index: int, video_key: str, chunk_size: int = None) -> Path:
        """获取视频文件路径"""
        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        episode_chunk = episode_index // chunk_size
        return self.dataset_path / self.VIDEO_PATH_TEMPLATE.format(
            episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index
        )

    def get_info_path(self) -> Path:
        """获取info.json路径"""
        return self.dataset_path / self.INFO_PATH

    def get_episodes_path(self) -> Path:
        """获取episodes.jsonl路径"""
        return self.dataset_path / self.EPISODES_PATH

    def get_stats_path(self) -> Path:
        """获取episodes_stats.jsonl路径"""
        return self.dataset_path / self.STATS_PATH

    def get_frames_path(self) -> Path:
        """获取frames.jsonl路径"""
        return self.dataset_path / self.FRAMES_PATH


class AppConfig:
    """应用配置管理"""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "app": {
                "host": "127.0.0.1",
                "port": 9090,
                "debug": False
            },
            "dataset": {
                "path": "/home/chenqingyu/robot/lerobot_data/datasets/grasp_dataset",
                "cache_enabled": False
            },
            "ui": {
                "episodes_per_page": 50,
                "default_video_keys": ["laptop", "phone"],
                "annotation": {
                    "autosave_on_next": True,
                    "write_mode": "append",
                    "output_format": "jsonl",
                    "default_skills": ["Pick", "Place", "Move", "Rotate", "Push", "Pull", "Grasp", "Release"],
                    "persist_dataset_in_session": True
                }
            },
            "motion_detection": {
                "enabled": True,
                "method": "frame_diff",
                "threshold": 0.02,
                "min_static_frames": 5,
                "resize_width": 320,
                "resize_height": 240,
                "gaussian_blur_kernel": 5,
                "cache_results": True,
                "cache_duration_hours": 24
            }
        }

    def get(self, key: str, default=None):
        """获取配置值 (支持点号路径)"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置值 (支持点号路径)"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self):
        """保存配置到文件"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

    def get_dataset_path(self) -> str:
        """获取数据集路径"""
        return self.get("dataset.path")

    def get_app_host(self) -> str:
        """获取应用主机"""
        return self.get("app.host", "127.0.0.1")

    def get_app_port(self) -> int:
        """获取应用端口"""
        return self.get("app.port", 9090)

    def get_debug_mode(self) -> bool:
        """获取调试模式"""
        return self.get("app.debug", False)

    def get_episodes_per_page(self) -> int:
        """获取每页episode数量"""
        return self.get("ui.episodes_per_page", 50)

    def get_default_video_keys(self) -> list:
        """获取默认视频键列表"""
        return self.get("ui.default_video_keys", ["laptop", "phone"])

    def is_cache_enabled(self) -> bool:
        """是否启用缓存"""
        return self.get("dataset.cache_enabled", False)

    def get_annotation_config(self, key: str = None):
        """获取标注配置"""
        if key:
            return self.get(f"ui.annotation.{key}")
        return self.get("ui.annotation", {})

    def get_motion_detection_config(self, key: str = None):
        """获取静止帧检测配置"""
        if key:
            return self.get(f"motion_detection.{key}")
        return self.get("motion_detection", {})

    def is_motion_detection_enabled(self) -> bool:
        """是否启用静止帧检测"""
        return self.get("motion_detection.enabled", True)

    def get_motion_detection_method(self) -> str:
        """获取静止帧检测方法"""
        return self.get("motion_detection.method", "frame_diff")

    def get_motion_detection_threshold(self) -> float:
        """获取静止帧检测阈值"""
        return self.get("motion_detection.threshold", 0.02)