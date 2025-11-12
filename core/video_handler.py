"""
视频处理模块
功能: 视频路径解析、文件验证、播放器配置、静止帧检测
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from .config import DatasetConfig, AppConfig
from .data_loader import LocalDatasetLoader
from .motion_detector import (
    MotionDetector,
    VideoMotionAnalyzer,
    MotionDetectionConfig,
    MotionDetectionMethod,
    MotionDetectionResult
)


class VideoHandler:
    """视频文件处理器"""

    SUPPORTED_FORMATS = ['.mp4', '.webm', '.avi']

    def __init__(self, dataset_config: DatasetConfig, data_loader: LocalDatasetLoader, app_config: Optional[AppConfig] = None):
        self.config = dataset_config
        self.data_loader = data_loader
        self.app_config = app_config
        # 动态获取视频键，而不是硬编码
        self._video_keys = self.data_loader.get_video_keys()

        # 初始化静止帧检测
        self._motion_analyzer = None
        self._motion_cache = {}  # 缓存检测结果
        self._init_motion_detection()

    def get_video_paths(self, episode_index: int, video_keys: List[str] = None) -> Dict[str, str]:
        """
        获取episode的视频文件路径
        返回格式: {"laptop": "path/to/laptop/video.mp4", "phone": "path/to/phone/video.mp4"}
        """
        video_keys = video_keys or self._video_keys
        video_paths = {}

        for video_key in video_keys:
            video_path = self.config.get_video_path(episode_index, video_key)
            if video_path.exists():
                # 转换为Web可访问的相对路径，去掉videos/前缀
                relative_path = video_path.relative_to(self.config.dataset_path)
                # 去掉videos/前缀，因为Flask路由会添加
                relative_path_str = str(relative_path)
                if relative_path_str.startswith('videos/'):
                    relative_path_str = relative_path_str[7:]  # 去掉 "videos/" 前缀
                # 使用简化的键名（去掉observation.images.前缀）
                simple_key = video_key.replace('observation.images.', '') if 'observation.images.' in video_key else video_key
                video_paths[simple_key] = relative_path_str

        return video_paths

    def validate_video_file(self, video_path: Path) -> bool:
        """验证视频文件是否有效"""
        if not video_path.exists():
            return False

        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False

        # 检查文件大小 (避免空文件)
        if video_path.stat().st_size == 0:
            return False

        return True

    def get_video_info(self, episode_index: int, video_keys: List[str] = None) -> List[Dict[str, str]]:
        """
        获取episode视频信息 (用于前端播放器配置)
        返回格式: [{"key": "laptop", "path": "relative/path", "type": "video/mp4"}, ...]
        """
        video_keys = video_keys or self._video_keys
        video_paths = self.get_video_paths(episode_index, video_keys)
        video_info = []

        for simple_key, relative_path in video_paths.items():
            video_info.append({
                "key": simple_key,
                "path": relative_path,
                "url": f"/videos/{relative_path}",
                "type": "video/mp4",
                "filename": simple_key
            })

        return video_info

    def get_all_video_keys(self, episode_index: int) -> List[str]:
        """获取episode所有可用的视频键"""
        available_keys = []

        for video_key in self._video_keys:
            video_path = self.config.get_video_path(episode_index, video_key)
            if video_path.exists() and self.validate_video_file(video_path):
                available_keys.append(video_key)

        return available_keys

    def check_video_availability(self, episode_index: int) -> Dict[str, bool]:
        """检查各个视频键的可用性"""
        availability = {}

        for video_key in self._video_keys:
            video_path = self.config.get_video_path(episode_index, video_key)
            availability[video_key] = video_path.exists() and self.validate_video_file(video_path)

        return availability

    def get_video_stats(self, episode_index: int) -> Dict[str, Any]:
        """获取视频统计信息"""
        stats = {
            "total_videos": 0,
            "available_videos": 0,
            "video_keys": [],
            "missing_keys": []
        }

        for video_key in self._video_keys:
            stats["total_videos"] += 1
            video_path = self.config.get_video_path(episode_index, video_key)

            if video_path.exists() and self.validate_video_file(video_path):
                stats["available_videos"] += 1
                stats["video_keys"].append(video_key)
            else:
                stats["missing_keys"].append(video_key)

        return stats

    def _init_motion_detection(self):
        """初始化静止帧检测"""
        if self.app_config and self.app_config.is_motion_detection_enabled():
            # 从配置创建检测配置
            method_str = self.app_config.get_motion_detection_method()
            method = MotionDetectionMethod(method_str)

            motion_config = MotionDetectionConfig(
                method=method,
                threshold=self.app_config.get_motion_detection_threshold(),
                min_static_frames=self.app_config.get_motion_detection_config("min_static_frames") or 5,
                resize_width=self.app_config.get_motion_detection_config("resize_width") or 320,
                resize_height=self.app_config.get_motion_detection_config("resize_height") or 240,
                gaussian_blur_kernel=self.app_config.get_motion_detection_config("gaussian_blur_kernel") or 5
            )

            self._motion_analyzer = VideoMotionAnalyzer(motion_config)

    def _get_motion_analyzer(self, threshold: Optional[float] = None, method: Optional[str] = None,
                           min_static_frames: Optional[int] = None) -> Optional[VideoMotionAnalyzer]:
        """获取或创建motion analyzer（支持动态参数）"""
        # 如果没有提供动态参数，使用默认的analyzer
        if threshold is None and method is None and min_static_frames is None:
            return self._motion_analyzer

        # 如果没有启用motion detection，返回None
        if not self.app_config or not self.app_config.is_motion_detection_enabled():
            return None

        # 创建动态配置
        from .static_frame_processor import MotionDetectionMethod

        # 使用提供的参数或默认值
        actual_method = method or self.app_config.get_motion_detection_method()
        actual_threshold = threshold if threshold is not None else self.app_config.get_motion_detection_threshold()
        actual_min_static_frames = min_static_frames if min_static_frames is not None else (
            self.app_config.get_motion_detection_config("min_static_frames") or 5
        )

        motion_config = MotionDetectionConfig(
            method=MotionDetectionMethod(actual_method),
            threshold=actual_threshold,
            min_static_frames=actual_min_static_frames,
            resize_width=self.app_config.get_motion_detection_config("resize_width") or 320,
            resize_height=self.app_config.get_motion_detection_config("resize_height") or 240,
            gaussian_blur_kernel=self.app_config.get_motion_detection_config("gaussian_blur_kernel") or 5
        )

        return VideoMotionAnalyzer(motion_config)

    def _get_cache_key(self, episode_index: int, video_key: str) -> str:
        """生成缓存键"""
        return f"{episode_index}_{video_key}"

    def _get_cache_key_with_params(self, episode_index: int, video_key: str,
                                 threshold: Optional[float] = None, method: Optional[str] = None,
                                 min_static_frames: Optional[int] = None) -> str:
        """生成包含检测参数的缓存键"""
        # 获取实际使用的参数值
        actual_method = method or (self.app_config.get_motion_detection_method() if self.app_config else "frame_diff")
        actual_threshold = threshold if threshold is not None else (
            self.app_config.get_motion_detection_threshold() if self.app_config else 0.02
        )
        actual_min_static_frames = min_static_frames if min_static_frames is not None else (
            self.app_config.get_motion_detection_config("min_static_frames") if self.app_config else 5
        )

        # 创建包含参数的缓存键
        return f"{episode_index}_{video_key}_{actual_method}_{actual_threshold:.4f}_{actual_min_static_frames}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """检查缓存是否有效"""
        if not self.app_config or not self.app_config.get_motion_detection_config("cache_results"):
            return False

        cache_duration_hours = self.app_config.get_motion_detection_config("cache_duration_hours") or 24
        cache_time = cache_entry.get("timestamp", 0)
        current_time = time.time()

        return (current_time - cache_time) < (cache_duration_hours * 3600)

    def detect_static_frames(self, episode_index: int, video_key: str,
                           start_frame: int = 0, end_frame: Optional[int] = None,
                           threshold: Optional[float] = None, method: Optional[str] = None,
                           min_static_frames: Optional[int] = None) -> List[MotionDetectionResult]:
        """检测视频中的静止帧"""
        # 创建动态配置或使用默认配置
        motion_analyzer = self._get_motion_analyzer(threshold, method, min_static_frames)
        if not motion_analyzer:
            return []

        # 检查缓存（包含动态参数）
        cache_key = self._get_cache_key_with_params(episode_index, video_key, threshold, method, min_static_frames)
        if cache_key in self._motion_cache and self._is_cache_valid(self._motion_cache[cache_key]):
            cached_results = self._motion_cache[cache_key]["results"]
            # 过滤结果范围
            if start_frame > 0 or end_frame is not None:
                filtered_results = []
                for result in cached_results:
                    if result.frame_index >= start_frame:
                        if end_frame is None or result.frame_index < end_frame:
                            filtered_results.append(result)
                return filtered_results
            return cached_results

        # 获取视频路径
        video_path = self.config.get_video_path(episode_index, video_key)
        if not video_path.exists():
            return []

        try:
            # 执行检测
            results = motion_analyzer.analyze_video(video_path, start_frame, end_frame)

            # 缓存结果
            if self.app_config and self.app_config.get_motion_detection_config("cache_results"):
                self._motion_cache[cache_key] = {
                    "results": results,
                    "timestamp": time.time()
                }

            return results
        except Exception as e:
            print(f"Error detecting static frames for {video_key} episode {episode_index}: {e}")
            return []

    def get_static_segments(self, episode_index: int, video_key: str,
                          threshold: Optional[float] = None, method: Optional[str] = None,
                          min_static_frames: Optional[int] = None) -> List[Tuple[int, int]]:
        """获取静止片段"""
        # 获取或创建motion analyzer
        motion_analyzer = self._get_motion_analyzer(threshold, method, min_static_frames)
        if not motion_analyzer:
            return []

        results = self.detect_static_frames(episode_index, video_key, threshold=threshold, method=method, min_static_frames=min_static_frames)
        return motion_analyzer.find_static_segments(results)

    def get_motion_statistics(self, episode_index: int, video_key: str,
                            threshold: Optional[float] = None, method: Optional[str] = None,
                            min_static_frames: Optional[int] = None) -> Dict[str, Any]:
        """获取运动统计信息"""
        # 获取或创建motion analyzer
        motion_analyzer = self._get_motion_analyzer(threshold, method, min_static_frames)
        if not motion_analyzer:
            return {}

        results = self.detect_static_frames(episode_index, video_key, threshold=threshold, method=method, min_static_frames=min_static_frames)
        return motion_analyzer.get_motion_statistics(results)

    def clear_motion_cache(self):
        """清除运动检测缓存"""
        self._motion_cache.clear()

    def get_motion_detection_status(self) -> Dict[str, Any]:
        """获取静止帧检测状态"""
        return {
            "enabled": self._motion_analyzer is not None,
            "cache_size": len(self._motion_cache),
            "method": self._motion_analyzer.config.method.value if self._motion_analyzer else None,
            "threshold": self._motion_analyzer.config.threshold if self._motion_analyzer else None
        }