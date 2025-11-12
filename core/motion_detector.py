"""
静止帧检测模块
功能: 运动检测算法、静止帧分析、片段规划
从原 static_frame_processor.py 中提取检测相关功能
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MotionDetectionMethod(Enum):
    """运动检测方法枚举"""
    FRAME_DIFF = "frame_diff"           # 帧差法


@dataclass
class MotionDetectionConfig:
    """运动检测配置"""
    method: MotionDetectionMethod = MotionDetectionMethod.FRAME_DIFF
    threshold: float = 0.001             # 静止阈值
    min_static_frames: int = 50          # 最小连续静止帧数
    resize_width: int = 320             # 处理时的图像宽度（提高性能）
    resize_height: int = 240            # 处理时的图像高度
    gaussian_blur_kernel: int = 5       # 高斯模糊核大小


@dataclass
class MotionDetectionResult:
    """运动检测结果"""
    frame_index: int
    is_static: bool
    motion_score: float
    timestamp: Optional[float] = None


class MotionDetector:
    """静止帧检测器"""

    def __init__(self, config: MotionDetectionConfig):
        self.config = config
        self.previous_frame = None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧：调整大小、转灰度、模糊"""
        # 调整大小以提高处理速度
        resized = cv2.resize(frame, (self.config.resize_width, self.config.resize_height))

        # 转换为灰度图
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # 高斯模糊减少噪声
        if self.config.gaussian_blur_kernel > 0:
            gray = cv2.GaussianBlur(gray,
                                  (self.config.gaussian_blur_kernel, self.config.gaussian_blur_kernel),
                                  0)

        return gray

    def detect_motion_frame_diff(self, current_frame: np.ndarray) -> float:
        """使用帧差法检测运动"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return 1.0  # 第一帧默认为运动

        # 计算帧差
        diff = cv2.absdiff(self.previous_frame, current_frame)

        # 计算运动分数（归一化的平均差值）
        motion_score = np.mean(diff) / 255.0

        # 更新前一帧
        self.previous_frame = current_frame.copy()

        return motion_score

    def detect_motion(self, frame: np.ndarray) -> float:
        """检测单帧的运动分数"""
        processed_frame = self.preprocess_frame(frame)
        return self.detect_motion_frame_diff(processed_frame)

    def is_static_frame(self, motion_score: float) -> bool:
        """判断是否为静止帧"""
        return motion_score < self.config.threshold

    def reset(self):
        """重置检测器状态"""
        self.previous_frame = None


class VideoMotionAnalyzer:
    """视频运动分析器"""

    def __init__(self, config: MotionDetectionConfig):
        self.config = config
        self.detector = MotionDetector(config)

    def analyze_video(self, video_path: Path,
                     start_frame: int = 0,
                     end_frame: Optional[int] = None) -> List[MotionDetectionResult]:
        """分析视频中的运动"""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        results = []
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # 跳转到起始帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 检查是否达到结束帧
                if end_frame is not None and frame_index >= end_frame:
                    break

                # 检测运动
                motion_score = self.detector.detect_motion(frame)
                is_static = self.detector.is_static_frame(motion_score)
                timestamp = frame_index / fps

                result = MotionDetectionResult(
                    frame_index=frame_index,
                    is_static=is_static,
                    motion_score=motion_score,
                    timestamp=timestamp
                )
                results.append(result)

                frame_index += 1

        finally:
            cap.release()

        return results

    def find_static_segments(self, results: List[MotionDetectionResult]) -> List[Tuple[int, int]]:
        """查找静止片段（连续的静止帧）"""
        segments = []
        current_start = None

        for result in results:
            if result.is_static:
                if current_start is None:
                    current_start = result.frame_index
            else:
                if current_start is not None:
                    # 检查片段长度是否满足最小要求
                    segment_length = result.frame_index - current_start
                    if segment_length >= self.config.min_static_frames:
                        segments.append((current_start, result.frame_index - 1))
                    current_start = None

        # 处理视频结尾的静止片段
        if current_start is not None and results:
            segment_length = results[-1].frame_index - current_start + 1
            if segment_length >= self.config.min_static_frames:
                segments.append((current_start, results[-1].frame_index))

        return segments

    def get_motion_statistics(self, results: List[MotionDetectionResult]) -> Dict[str, Any]:
        """获取运动统计信息"""
        if not results:
            return {
                "total_frames": 0,
                "static_frames": 0,
                "motion_frames": 0,
                "static_ratio": 0.0,
                "avg_motion_score": 0.0,
                "max_motion_score": 0.0,
                "min_motion_score": 0.0
            }

        static_frames = sum(1 for r in results if r.is_static)
        motion_frames = len(results) - static_frames
        motion_scores = [r.motion_score for r in results]

        return {
            "total_frames": len(results),
            "static_frames": static_frames,
            "motion_frames": motion_frames,
            "static_ratio": static_frames / len(results),
            "avg_motion_score": np.mean(motion_scores),
            "max_motion_score": np.max(motion_scores),
            "min_motion_score": np.min(motion_scores)
        }

    def combine_detection_results(self, results_by_view: Dict[str, List[MotionDetectionResult]], 
                                 policy: str = "intersection") -> List[int]:
        """
        合并多视角的检测结果，生成统一的静止帧索引列表
        
        Args:
            results_by_view: 各视角的检测结果，键为视角名，值为检测结果列表
            policy: 合并策略，"intersection"（交集，保守）或"union"（并集，激进）
        
        Returns:
            统一的静止帧索引列表（倒序排序）
        """
        if not results_by_view:
            return []
        
        # 获取所有视角的静止帧索引集合
        static_frames_by_view = {}
        for view_name, results in results_by_view.items():
            static_frames = {r.frame_index for r in results if r.is_static}
            static_frames_by_view[view_name] = static_frames
        
        if not static_frames_by_view:
            return []
        
        # 根据策略合并
        if policy == "intersection":
            # 取交集：所有视角都认为是静止帧的帧
            combined_static = set.intersection(*static_frames_by_view.values())
        elif policy == "union":
            # 取并集：任一视角认为是静止帧的帧
            combined_static = set.union(*static_frames_by_view.values())
        else:
            raise ValueError(f"Unknown combine policy: {policy}")
        
        # 转换为倒序排序的列表
        return sorted(list(combined_static), reverse=True)
