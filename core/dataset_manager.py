"""
数据集管理模块
功能: 删除episode、备份数据集、元数据更新、基于技能的数据集分割
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, NamedTuple, Tuple
import logging
import time
import cv2
import numpy as np
import fcntl
import hashlib
import subprocess
import tempfile
from datetime import datetime
from collections import OrderedDict
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from PIL import Image as PILImage

from .motion_detector import (
    MotionDetectionConfig, MotionDetectionResult,
    VideoMotionAnalyzer, MotionDetectionMethod
)
from .config import DatasetConfig
from .annotation_store import AnnotationStore


class EpisodeSegment(NamedTuple):
    """Episode片段信息"""
    episode_id: int
    skill: str
    start_frame: int
    end_frame: int
    action_text: str
    length: int


class ValidationReport(NamedTuple):
    """验证报告"""
    success: bool
    errors: List[str]
    warnings: List[str]


def load_json_lines(filepath: Path) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json_lines(data: List[Dict], filepath: Path):
    """保存JSONL文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


class DatasetManager:
    """
    数据集管理器
    负责删除episodes、创建备份、更新元数据等操作
    """

    def __init__(self, dataset_path: str, app_config=None):
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        self.app_config = app_config

        # 验证数据集路径
        if not self.dataset_path.exists():
            raise ValueError(f"数据集路径不存在: {dataset_path}")

        if not (self.dataset_path / "meta" / "info.json").exists():
            raise ValueError(f"不是有效的LeRobot数据集: {dataset_path}")

        # 初始化静止帧删除服务
        self.static_frames = StaticFrameDeletionService(dataset_path, app_config)

        # 初始化数据集配置和标注存储
        self.dataset_config = DatasetConfig(dataset_path)
        self.annotation_store = AnnotationStore(self.dataset_config)

        # 分割相关配置
        self.fps = 30  # 默认FPS，将从info.json中读取

    def detect_camera_names(self) -> List[str]:
        """检测数据集中的摄像头名称"""
        camera_names = set()
        video_chunk_dir = self.dataset_path / "videos" / "chunk-000"

        if video_chunk_dir.exists():
            for item in video_chunk_dir.iterdir():
                if item.is_dir() and item.name.startswith("observation.images."):
                    camera_name = item.name.replace("observation.images.", "")
                    camera_names.add(camera_name)

        return sorted(list(camera_names))

    def get_delete_preview(self, episode_id: int) -> Dict[str, Any]:
        """获取删除预览信息"""
        try:
            # 加载元数据
            with open(self.dataset_path / "meta" / "info.json", 'r') as f:
                info = json.load(f)

            # 验证episode是否存在
            if episode_id < 0 or episode_id >= info['total_episodes']:
                return {
                    "error": f"Episode {episode_id} 不存在。有效范围: 0-{info['total_episodes']-1}"
                }

            # 统计要删除的数据
            frames_to_remove = 0
            videos_to_remove = 0
            cameras = self.detect_camera_names()

            # 统计帧数
            parquet_file = self.dataset_path / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                frames_to_remove = len(df)

            # 统计视频数
            for camera in cameras:
                video_file = self.dataset_path / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{episode_id:06d}.mp4"
                if video_file.exists():
                    videos_to_remove += 1

            # 检查备份状态
            backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_backup"
            backup_exists = backup_path.exists()

            return {
                "episode_id": episode_id,
                "current_stats": {
                    "total_episodes": info['total_episodes'],
                    "total_frames": info['total_frames'],
                    "total_videos": info['total_videos']
                },
                "to_delete": {
                    "episodes": 1,
                    "frames": frames_to_remove,
                    "videos": videos_to_remove,
                    "cameras": cameras
                },
                "after_delete": {
                    "total_episodes": info['total_episodes'] - 1,
                    "total_frames": info['total_frames'] - frames_to_remove,
                    "total_videos": info['total_videos'] - videos_to_remove
                },
                "backup": {
                    "path": str(backup_path),
                    "exists": backup_exists
                }
            }

        except Exception as e:
            self.logger.error(f"获取删除预览失败: {e}")
            return {"error": str(e)}

    def create_backup(self) -> Dict[str, Any]:
        """创建数据集备份"""
        backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_backup"

        try:
            if backup_path.exists():
                return {
                    "success": True,
                    "message": f"备份目录已存在，跳过备份",
                    "backup_path": str(backup_path),
                    "already_exists": True
                }

            self.logger.info(f"创建数据集备份: {backup_path}")
            shutil.copytree(self.dataset_path, backup_path)

            return {
                "success": True,
                "message": f"备份创建成功",
                "backup_path": str(backup_path),
                "already_exists": False
            }

        except Exception as e:
            self.logger.error(f"创建备份失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def delete_episode(self, episode_id: int, create_backup: bool = True) -> Dict[str, Any]:
        """删除指定episode"""
        try:
            # 获取删除预览信息
            preview = self.get_delete_preview(episode_id)
            if "error" in preview:
                return {"success": False, "error": preview["error"]}

            # 创建备份（如果需要）
            if create_backup:
                backup_result = self.create_backup()
                if not backup_result["success"]:
                    return {
                        "success": False,
                        "error": f"备份创建失败: {backup_result['error']}"
                    }

            # 执行删除操作
            result = self._perform_delete(episode_id)

            # 添加预览信息到结果中
            result["preview"] = preview
            if create_backup:
                result["backup"] = backup_result

            return result

        except Exception as e:
            self.logger.error(f"删除episode {episode_id} 失败: {e}")
            return {"success": False, "error": str(e)}

    def _perform_delete(self, episode_id: int) -> Dict[str, Any]:
        """执行实际的删除操作"""
        try:
            # 加载元数据
            with open(self.dataset_path / "meta" / "info.json", 'r') as f:
                info = json.load(f)

            episodes = load_json_lines(self.dataset_path / "meta" / "episodes.jsonl")
            episodes_stats = load_json_lines(self.dataset_path / "meta" / "episodes_stats.jsonl")

            episodes_to_remove = {episode_id}

            # 统计删除数据
            frames_to_remove = 0
            videos_to_remove = 0

            # 删除Parquet文件
            parquet_file = self.dataset_path / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                frames_to_remove = len(df)
                parquet_file.unlink()

            # 删除视频文件
            camera_names = self.detect_camera_names()
            for camera in camera_names:
                video_file = self.dataset_path / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{episode_id:06d}.mp4"
                if video_file.exists():
                    video_file.unlink()
                    videos_to_remove += 1

            # 重新编号剩余episodes
            remaining_episodes = sorted([i for i in range(info['total_episodes']) if i not in episodes_to_remove])

            # 重命名Parquet文件并更新索引
            temp_files = []
            for new_idx, old_idx in enumerate(remaining_episodes):
                old_file = self.dataset_path / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet"
                temp_file = self.dataset_path / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet.tmp"

                if old_file.exists():
                    old_file.rename(temp_file)
                    temp_files.append((temp_file, new_idx))

            # 重命名为最终文件名并更新索引
            global_index_start = 0
            for temp_file, new_idx in temp_files:
                final_file = self.dataset_path / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"

                df = pd.read_parquet(temp_file)
                df['episode_index'] = new_idx
                df['index'] = range(global_index_start, global_index_start + len(df))
                df.to_parquet(final_file, index=False)

                temp_file.unlink()
                global_index_start += len(df)

            # 重命名视频文件
            for camera in camera_names:
                temp_videos = []
                for new_idx, old_idx in enumerate(remaining_episodes):
                    old_video = self.dataset_path / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{old_idx:06d}.mp4"
                    temp_video = self.dataset_path / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{old_idx:06d}.mp4.tmp"

                    if old_video.exists():
                        old_video.rename(temp_video)
                        temp_videos.append((temp_video, new_idx))

                for temp_video, new_idx in temp_videos:
                    final_video = self.dataset_path / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{new_idx:06d}.mp4"
                    temp_video.rename(final_video)

            # 更新元数据
            new_episodes = []
            for new_idx, old_idx in enumerate(remaining_episodes):
                episode_data = episodes[old_idx].copy()
                episode_data['episode_index'] = new_idx
                new_episodes.append(episode_data)

            new_episodes_stats = []
            for new_idx, old_idx in enumerate(remaining_episodes):
                stats_data = episodes_stats[old_idx].copy()
                stats_data['episode_index'] = new_idx
                new_episodes_stats.append(stats_data)

            # 更新info.json
            new_total_episodes = len(remaining_episodes)
            new_total_frames = info['total_frames'] - frames_to_remove
            new_total_videos = info['total_videos'] - videos_to_remove

            info['total_episodes'] = new_total_episodes
            info['total_frames'] = new_total_frames
            info['total_videos'] = new_total_videos
            info['splits']['train'] = f"0:{new_total_episodes}"

            # 保存更新后的元数据
            with open(self.dataset_path / "meta" / "info.json", 'w') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            save_json_lines(new_episodes, self.dataset_path / "meta" / "episodes.jsonl")
            save_json_lines(new_episodes_stats, self.dataset_path / "meta" / "episodes_stats.jsonl")

            # 计算建议的重定向episode
            next_episode_id = episode_id if episode_id < new_total_episodes else max(0, new_total_episodes - 1)

            return {
                "success": True,
                "deleted": {
                    "episode_id": episode_id,
                    "frames": frames_to_remove,
                    "videos": videos_to_remove
                },
                "final_stats": {
                    "total_episodes": new_total_episodes,
                    "total_frames": new_total_frames,
                    "total_videos": new_total_videos
                },
                "redirect_to": next_episode_id if new_total_episodes > 0 else None
            }

        except Exception as e:
            self.logger.error(f"执行删除操作失败: {e}")
            return {"success": False, "error": str(e)}

    def split_by_skill(self, output_root: str) -> Dict[str, Any]:
        """基于技能分割数据集 - 自动提取所有技能"""
        try:
            self.logger.info(f"开始基于技能的数据分割，输出目录: {output_root}")

            # 从AnnotationStore加载所有标注数据
            all_annotations = self.annotation_store.load_all()

            if not all_annotations:
                return {"success": False, "error": "未找到任何标注数据"}

            # 提取所有技能和片段
            segments, available_skills = self._extract_segments_from_annotations(all_annotations)

            if not available_skills:
                return {"success": False, "error": "未找到任何技能标注"}

            self.logger.info(f"发现 {len(available_skills)} 种技能: {available_skills}")
            self.logger.info(f"共 {len(segments)} 个片段")

            # 显示每个技能的片段统计
            skill_counts = {}
            for segment in segments:
                skill_counts[segment.skill] = skill_counts.get(segment.skill, 0) + 1

            self.logger.info("技能片段统计:")
            for skill in sorted(skill_counts.keys()):
                self.logger.info(f"  {skill}: {skill_counts[skill]} 个片段")

            # 加载FPS信息
            self._load_fps()

            # 处理每个技能
            processed_skills = []
            failed_skills = []

            for skill in available_skills:
                try:
                    self.logger.info(f"开始处理技能: {skill}")
                    self._process_skill(skill, segments, output_root)
                    processed_skills.append(skill)
                    self.logger.info(f"技能 {skill} 处理完成")
                except Exception as e:
                    self.logger.error(f"处理技能 {skill} 时发生错误: {e}")
                    failed_skills.append(skill)
                    continue

            result = {
                "success": True,
                "processed_skills": processed_skills,
                "failed_skills": failed_skills,
                "total_skills": len(available_skills),
                "output_root": output_root
            }

            if failed_skills:
                result["warning"] = f"部分技能处理失败: {failed_skills}"

            self.logger.info(f"数据集分割完成，成功处理: {processed_skills}")
            return result

        except Exception as e:
            self.logger.error(f"数据集分割失败: {e}")
            return {"success": False, "error": str(e)}

    def _extract_segments_from_annotations(self, all_annotations: Dict) -> Tuple[List[EpisodeSegment], List[str]]:
        """从标注数据中提取片段和技能列表"""
        segments = []
        skills_set = set()

        for episode_id, annotation_data in all_annotations.items():
            action_config = annotation_data.get('label_info', {}).get('action_config', [])

            for action in action_config:
                start_frame = action['start_frame']
                end_frame = action['end_frame']
                skill = action['skill']
                action_text = action['action_text']
                length = end_frame - start_frame + 1

                skills_set.add(skill)

                segment = EpisodeSegment(
                    episode_id=int(episode_id),
                    skill=skill,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    action_text=action_text,
                    length=length
                )
                segments.append(segment)

        return segments, sorted(list(skills_set))

    def _load_fps(self):
        """加载视频帧率"""
        try:
            info_path = self.dataset_path / "meta" / "info.json"
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            self.fps = info.get('fps', 30.0)
        except Exception as e:
            self.logger.warning(f"无法加载fps，使用默认值30.0: {e}")
            self.fps = 30.0

    def _process_skill(self, skill: str, segments: List[EpisodeSegment], output_root: str):
        """处理单个技能的所有片段"""
        # 获取该技能的所有片段
        skill_segments = [seg for seg in segments if seg.skill == skill]
        skill_segments.sort(key=lambda x: x.episode_id)

        if not skill_segments:
            self.logger.warning(f"技能 {skill} 没有找到任何片段")
            return

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(output_root) / f"koch__skill-{skill.lower()}-{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"输出目录: {out_dir}")

        episodes_meta = []

        # 处理每个片段
        for new_idx, segment in enumerate(skill_segments):
            self.logger.info(f"处理片段 {new_idx + 1}/{len(skill_segments)}: "
                           f"Episode {segment.episode_id}, {segment.skill}, "
                           f"帧 {segment.start_frame}-{segment.end_frame}")

            try:
                # 处理Parquet数据
                src_parquet = self.dataset_config.get_data_path(segment.episode_id)
                table = self._cut_parquet(src_parquet, segment.start_frame, segment.end_frame,
                                       new_idx, skill, segment)
                out_parquet = out_dir / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
                self._write_parquet(table, out_parquet)

                # 处理视频文件
                video_keys = ["observation.images.laptop", "observation.images.phone"]
                t_start = segment.start_frame / self.fps
                t_end_exclusive = segment.end_frame / self.fps

                for video_key in video_keys:
                    src_video = self.dataset_config.get_video_path(segment.episode_id, video_key)
                    out_video = out_dir / "videos" / "chunk-000" / video_key / f"episode_{new_idx:06d}.mp4"

                    success = self._cut_video_ffmpeg(src_video, t_start, t_end_exclusive, out_video)
                    if not success:
                        self.logger.error(f"视频切割失败: {src_video} -> {out_video}")
                        continue

                # 记录episode元数据
                actual_length = segment.length - 1
                episode_meta = {
                    "episode_index": new_idx,
                    "tasks": [skill],
                    "length": actual_length
                }
                episodes_meta.append(episode_meta)

            except Exception as e:
                self.logger.error(f"处理片段失败: {e}")
                continue

        # 生成元数据
        self._emit_meta(out_dir, episodes_meta, skill)

        self.logger.info(f"技能 {skill} 处理完成，共 {len(episodes_meta)} 个片段")

    def _cut_parquet(self, src_parquet: Path, start_frame: int, end_frame: int,
                   new_episode_index: int, skill: str, segment: EpisodeSegment) -> pa.Table:
        """切割Parquet数据文件"""
        if not src_parquet.exists():
            raise FileNotFoundError(f"源Parquet文件不存在: {src_parquet}")

        # 读取数据
        table = pq.read_table(src_parquet)

        # 调整end_frame逻辑：将end_frame减1，改为半开区间
        actual_end_frame = end_frame - 1
        self.logger.debug(f"调整end_frame: 原始{end_frame} -> 实际{actual_end_frame}")

        # 检查边界有效性
        max_frame_index = table['frame_index'].to_pylist()[-1]
        if actual_end_frame > max_frame_index:
            self.logger.warning(f"actual_end_frame({actual_end_frame})超出数据范围(max={max_frame_index})，调整为{max_frame_index}")
            actual_end_frame = max_frame_index

        # 过滤数据（使用调整后的end_frame）
        frame_index_col = table['frame_index']
        filter_condition = pc.and_(
            pc.greater_equal(frame_index_col, start_frame),
            pc.less_equal(frame_index_col, actual_end_frame)
        )
        filtered_table = table.filter(filter_condition)

        # 验证帧数（基于调整后的范围）
        expected_length = actual_end_frame - start_frame + 1
        actual_length = len(filtered_table)
        if actual_length != expected_length:
            self.logger.warning(f"帧数不匹配: 期望{expected_length}, 实际{actual_length}")

        # 重建字段
        new_data = {}

        # 复制所有原始列
        for col_name in table.column_names:
            new_data[col_name] = filtered_table[col_name]

        # 重建frame_index (从0开始)
        new_frame_indices = pa.array(range(len(filtered_table)), type=pa.int64())
        new_data['frame_index'] = new_frame_indices

        # 重建index (从0开始)
        if 'index' in table.column_names:
            new_data['index'] = new_frame_indices

        # 重建timestamp (从0开始)
        new_timestamps = pa.array([i / self.fps for i in range(len(filtered_table))], type=pa.float32())
        new_data['timestamp'] = new_timestamps

        # 更新episode_index
        new_data['episode_index'] = pa.array([new_episode_index] * len(filtered_table), type=pa.int64())

        # 更新task_index (动态生成)
        new_data['task_index'] = pa.array([0] * len(filtered_table), type=pa.int64())

        # 添加追溯字段
        new_data['original_episode_index'] = pa.array([segment.episode_id] * len(filtered_table), type=pa.int64())
        new_data['original_frame_index'] = filtered_table['frame_index']
        new_data['skill'] = pa.array([skill] * len(filtered_table), type=pa.string())
        new_data['action_text'] = pa.array([segment.action_text] * len(filtered_table), type=pa.string())

        # 保存原始timestamp
        if 'timestamp' in table.column_names:
            new_data['timestamp_original'] = filtered_table['timestamp']

        # 创建新表
        new_table = pa.table(new_data)

        self.logger.debug(f"Parquet切割完成: {len(filtered_table)} 行")
        return new_table

    def _write_parquet(self, table: pa.Table, out_path: Path):
        """写入Parquet文件"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        self.logger.debug(f"Parquet文件已保存: {out_path}")

    def _cut_video_ffmpeg(self, src_mp4: Path, t_start: float, t_end_exclusive: float,
                        out_mp4: Path, reencode_on_gop: bool = False) -> bool:
        """使用FFmpeg切割视频"""
        if not src_mp4.exists():
            raise FileNotFoundError(f"源视频文件不存在: {src_mp4}")

        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        # 首先尝试copy模式
        copy_cmd = [
            "ffmpeg", "-y", "-i", str(src_mp4),
            "-ss", f"{t_start:.6f}",
            "-to", f"{t_end_exclusive:.6f}",
            "-c", "copy",
            str(out_mp4)
        ]

        try:
            result = subprocess.run(copy_cmd, capture_output=True, text=True, check=True)
            self.logger.debug(f"视频切割成功 (copy模式): {out_mp4}")
            return True
        except subprocess.CalledProcessError as e:
            if reencode_on_gop:
                self.logger.warning(f"copy模式失败，尝试重编码模式: {e}")

                # 重编码模式
                reencode_cmd = [
                    "ffmpeg", "-y", "-i", str(src_mp4),
                    "-ss", f"{t_start:.6f}",
                    "-to", f"{t_end_exclusive:.6f}",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an",
                    str(out_mp4)
                ]

                try:
                    result = subprocess.run(reencode_cmd, capture_output=True, text=True, check=True)
                    self.logger.debug(f"视频切割成功 (重编码模式): {out_mp4}")
                    return True
                except subprocess.CalledProcessError as e2:
                    self.logger.error(f"视频切割失败 (重编码模式): {e2}")
                    return False
            else:
                self.logger.error(f"视频切割失败 (copy模式): {e}")
                return False

    def _emit_meta(self, out_root: Path, episodes_meta: List[Dict], skill: str):
        """生成元数据文件"""
        meta_dir = out_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # 生成tasks.jsonl
        tasks_file = meta_dir / "tasks.jsonl"
        with open(tasks_file, 'w', encoding='utf-8') as f:
            task_data = {"task_index": 0, "task": skill}
            f.write(json.dumps(task_data) + '\n')

        # 生成episodes.jsonl
        episodes_file = meta_dir / "episodes.jsonl"
        with open(episodes_file, 'w', encoding='utf-8') as f:
            for episode_meta in episodes_meta:
                f.write(json.dumps(episode_meta) + '\n')

        # 生成info.json
        self._generate_info_json(out_root, episodes_meta, skill)

        # 生成episodes_stats.jsonl
        self._generate_episodes_stats(out_root, episodes_meta)

        self.logger.info(f"元数据文件已生成: {meta_dir}")

    def _generate_info_json(self, out_root: Path, episodes_meta: List[Dict], skill: str):
        """生成info.json文件"""
        # 读取原始info.json作为模板
        source_info_path = self.dataset_config.get_info_path()
        with open(source_info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)

        # 更新统计信息
        total_episodes = len(episodes_meta)
        total_frames = sum(ep['length'] for ep in episodes_meta)
        total_videos = total_episodes * 2  # laptop + phone

        info_data.update({
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_videos": total_videos,
            "total_chunks": 1,  # 目前都在chunk-000
            "chunks_size": 1000,
            "splits": {"train": f"0:{total_episodes}"}
        })

        # 添加新字段的特征描述
        if "features" not in info_data:
            info_data["features"] = {}

        # 添加追溯字段的特征描述
        new_features = {
            "original_episode_index": {
                "dtype": "int64",
                "shape": [],
                "names": None
            },
            "original_frame_index": {
                "dtype": "int64",
                "shape": [],
                "names": None
            },
            "skill": {
                "dtype": "string",
                "shape": [],
                "names": None
            },
            "action_text": {
                "dtype": "string",
                "shape": [],
                "names": None
            },
            "timestamp_original": {
                "dtype": "float32",
                "shape": [],
                "names": None
            }
        }

        info_data["features"].update(new_features)

        # 保存info.json
        info_file = out_root / "meta" / "info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2)

    def _generate_episodes_stats(self, out_root: Path, episodes_meta: List[Dict]):
        """生成episodes_stats.jsonl文件"""
        stats_file = out_root / "meta" / "episodes_stats.jsonl"

        with open(stats_file, 'w', encoding='utf-8') as f:
            for episode_meta in episodes_meta:
                episode_index = episode_meta['episode_index']

                # 读取对应的Parquet文件计算真实统计信息
                parquet_file = out_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"

                if not parquet_file.exists():
                    self.logger.error(f"Parquet文件不存在: {parquet_file}")
                    continue

                # 计算真实的统计信息
                stats = self._calculate_episode_stats(parquet_file)

                stats_data = {
                    "episode_index": episode_index,
                    "stats": stats
                }
                f.write(json.dumps(stats_data) + '\n')

    def _calculate_episode_stats(self, parquet_file: Path) -> Dict[str, Any]:
        """计算单个episode的统计信息"""
        import numpy as np

        # 读取Parquet数据
        table = pq.read_table(parquet_file)
        episode_length = len(table)

        # 使用OrderedDict保持字段顺序
        stats = OrderedDict()

        # 1. action字段
        if 'action' in table.column_names:
            arrays = np.stack(table['action'].to_pylist())
            stats['action'] = {
                'min': np.min(arrays, axis=0).tolist(),
                'max': np.max(arrays, axis=0).tolist(),
                'mean': np.mean(arrays, axis=0).tolist(),
                'std': np.std(arrays, axis=0).tolist(),
                'count': [episode_length]
            }

        # 2. observation.state字段
        if 'observation.state' in table.column_names:
            arrays = np.stack(table['observation.state'].to_pylist())
            stats['observation.state'] = {
                'min': np.min(arrays, axis=0).tolist(),
                'max': np.max(arrays, axis=0).tolist(),
                'mean': np.mean(arrays, axis=0).tolist(),
                'std': np.std(arrays, axis=0).tolist(),
                'count': [episode_length]
            }

        # 3. 图像字段
        image_fields = ['observation.images.laptop', 'observation.images.phone']
        for field in image_fields:
            original_stats = self._get_image_field_stats(field)
            if original_stats:
                stats[field] = original_stats

        # 4. 标量字段
        scalar_fields = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
        for field in scalar_fields:
            if field in table.column_names:
                values = table[field].to_numpy()

                if field in ['frame_index', 'episode_index', 'index', 'task_index']:
                    # 整数字段
                    stats[field] = {
                        'min': [int(np.min(values))],
                        'max': [int(np.max(values))],
                        'mean': [float(np.mean(values))],
                        'std': [float(np.std(values))],
                        'count': [episode_length]
                    }
                else:
                    # 浮点数字段
                    stats[field] = {
                        'min': [float(np.min(values))],
                        'max': [float(np.max(values))],
                        'mean': [float(np.mean(values))],
                        'std': [float(np.std(values))],
                        'count': [episode_length]
                    }

        return dict(stats)

    def _get_image_field_stats(self, field: str) -> Dict[str, Any]:
        """获取图像字段的统计信息（从原始数据集中采样）"""
        try:
            original_stats_file = self.dataset_path / "meta" / "episodes_stats.jsonl"
            if original_stats_file.exists():
                with open(original_stats_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        original_stats = json.loads(first_line)
                        if field in original_stats['stats']:
                            return original_stats['stats'][field]
        except Exception as e:
            self.logger.warning(f"无法获取原始图像统计信息: {e}")

        return None


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10000, power: float = 0.75
) -> int:
    """根据数据集大小估算采样数量的启发式算法

    Args:
        dataset_len: 数据集长度
        min_num_samples: 最小采样数量
        max_num_samples: 最大采样数量
        power: 控制采样数量相对于数据集大小的增长率

    Returns:
        采样数量
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> List[int]:
    """生成均匀分布的采样索引列表

    Args:
        data_len: 数据长度

    Returns:
        采样索引列表
    """
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    """自动下采样图像以减少内存使用

    Args:
        img: 输入图像 (C, H, W)格式
        target_size: 目标尺寸
        max_size_threshold: 最大尺寸阈值

    Returns:
        下采样后的图像
    """
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # 不需要下采样
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def load_image_as_numpy(
    fpath: Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    """加载图像文件为numpy数组

    Args:
        fpath: 图像文件路径
        dtype: 数据类型
        channel_first: 是否使用通道优先格式 (C, H, W)

    Returns:
        图像数组
    """
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    return img_array


def sample_images(image_paths: List[Path]) -> np.ndarray:
    """采样图像并计算统计信息

    Args:
        image_paths: 图像文件路径列表

    Returns:
        采样的图像数组 (N, C, H, W)
    """
    if not image_paths:
        return np.array([])

    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        if not path.exists():
            continue

        # 加载为uint8以减少内存使用
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images if images is not None else np.array([])


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> Dict[str, np.ndarray]:
    """计算特征统计信息

    Args:
        array: 输入数组
        axis: 计算统计的轴
        keepdims: 是否保持维度

    Returns:
        包含min, max, mean, std, count的统计字典
    """
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_image_stats_from_video(video_path: Path, episode_length: int) -> Dict[str, np.ndarray]:
    """从视频文件计算图像统计信息

    Args:
        video_path: 视频文件路径
        episode_length: episode长度（帧数）

    Returns:
        图像统计信息字典
    """
    if not video_path.exists():
        # 返回默认统计值
        return {
            "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
            "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
            "mean": np.array([[[0.5]], [[0.5]], [[0.5]]]),
            "std": np.array([[[0.3]], [[0.3]], [[0.3]]]),
            "count": np.array([100]),
        }

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取采样帧索引
        sampled_indices = sample_indices(episode_length)

        images = []
        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # OpenCV读取的是BGR格式，转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为通道优先格式 (C, H, W)
            frame_chw = np.transpose(frame_rgb, (2, 0, 1)).astype(np.uint8)
            frame_chw = auto_downsample_height_width(frame_chw)
            images.append(frame_chw)

        cap.release()

        if not images:
            # 如果没有成功读取任何帧，返回默认值
            return {
                "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
                "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
                "mean": np.array([[[0.5]], [[0.5]], [[0.5]]]),
                "std": np.array([[[0.3]], [[0.3]], [[0.3]]]),
                "count": np.array([100]),
            }

        # 将图像列表转换为数组
        ep_ft_array = np.stack(images)  # (N, C, H, W)

        # 计算统计信息，保持通道维度
        axes_to_reduce = (0, 2, 3)  # 在batch, height, width维度上计算统计
        keepdims = True

        stats = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # 归一化并调整格式以匹配LeRobot格式
        normalized_stats = {
            k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()
        }

        return normalized_stats

    except Exception as e:
        logging.getLogger(__name__).warning(f"从视频计算图像统计失败: {e}")
        # 返回默认统计值
        return {
            "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
            "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
            "mean": np.array([[[0.5]], [[0.5]], [[0.5]]]),
            "std": np.array([[[0.3]], [[0.3]], [[0.3]]]),
            "count": np.array([100]),
        }


class StaticFrameDeletionService:
    """
    静止帧删除服务
    负责静止帧检测、计划制定、五层数据同步删除
    """

    def __init__(self, dataset_path: str, app_config=None):
        self.dataset_path = Path(dataset_path)
        self.app_config = app_config
        self.logger = logging.getLogger(__name__)

        # 加载配置
        self._load_config()

        # 初始化运动检测器
        self.motion_config = MotionDetectionConfig(
            method=MotionDetectionMethod.FRAME_DIFF,
            threshold=self.config.get('threshold', 0.001),
            min_static_frames=self.config.get('min_static_frames', 50),
            resize_width=self.config.get('resize_width', 320),
            resize_height=self.config.get('resize_height', 240),
            gaussian_blur_kernel=self.config.get('gaussian_blur_kernel', 5)
        )

        # 缓存
        self._motion_cache = {}
        self._video_fps = None
        self._video_keys = None

        # 确保锁目录存在
        self.locks_dir = self.dataset_path / ".locks"
        self.locks_dir.mkdir(exist_ok=True)

        # 确保操作日志目录存在
        self.ops_logs_dir = self.dataset_path / "meta" / "ops_logs"
        self.ops_logs_dir.mkdir(exist_ok=True)

    def _load_config(self):
        """加载配置"""
        self.config = {}
        if self.app_config and hasattr(self.app_config, 'motion_detection'):
            motion_config = self.app_config.motion_detection
            self.config = {
                'threshold': motion_config.get('threshold', 0.02),
                'min_static_frames': motion_config.get('min_static_frames', 5),
                'resize_width': motion_config.get('resize_width', 320),
                'resize_height': motion_config.get('resize_height', 240),
                'gaussian_blur_kernel': motion_config.get('gaussian_blur_kernel', 5),
                'combine_policy': motion_config.get('combine_policy', 'intersection'),
                'video_batch_size': motion_config.get('video_batch_size', 4)
            }

    def _load_fps(self) -> float:
        """加载视频帧率"""
        if self._video_fps is not None:
            return self._video_fps

        try:
            info_path = self.dataset_path / "meta" / "info.json"
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            self._video_fps = info.get('fps', 30.0)
        except Exception as e:
            self.logger.warning(f"无法加载fps，使用默认值30.0: {e}")
            self._video_fps = 30.0

        return self._video_fps

    def _load_video_keys(self) -> List[str]:
        """加载视频键列表"""
        if self._video_keys is not None:
            return self._video_keys

        try:
            info_path = self.dataset_path / "meta" / "info.json"
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)

            # 从features中提取视频键
            features = info.get('features', {})
            video_keys = []
            for key, feature_info in features.items():
                if isinstance(feature_info, dict) and feature_info.get('dtype') == 'video':
                    # 提取视频键的最后部分作为目录名
                    if '.' in key:
                        video_key = key.split('.')[-1]
                    else:
                        video_key = key
                    video_keys.append(video_key)

            self._video_keys = video_keys
        except Exception as e:
            self.logger.warning(f"无法加载视频键，使用默认值: {e}")
            self._video_keys = ['laptop', 'phone']

        return self._video_keys

    def _get_video_file_path(self, episode_id: int, video_key: str) -> Optional[Path]:
        """获取视频文件路径"""
        try:
            chunk_id = episode_id // 1000
            chunk_dir = f"chunk-{chunk_id:03d}"
            episode_filename = f"episode_{episode_id:06d}.mp4"

            # 处理不同格式的video_key
            if video_key.startswith("observation.images."):
                # 如果已经是完整的键，直接使用
                video_dir = video_key
            else:
                # 如果是短键（如laptop、phone），添加前缀
                video_dir = f"observation.images.{video_key}"

            video_path = self.dataset_path / "videos" / chunk_dir / video_dir / episode_filename

            if video_path.exists():
                return video_path
            else:
                self.logger.warning(f"视频文件不存在: {video_path}")
                return None
        except Exception as e:
            self.logger.error(f"获取视频路径失败: episode={episode_id}, video_key={video_key}, error={e}")
            return None

    def _get_parquet_file_path(self, episode_id: int) -> Optional[Path]:
        """获取Parquet文件路径"""
        try:
            chunk_id = episode_id // 1000
            chunk_dir = f"chunk-{chunk_id:03d}"
            episode_filename = f"episode_{episode_id:06d}.parquet"

            parquet_path = self.dataset_path / "data" / chunk_dir / episode_filename

            if parquet_path.exists():
                return parquet_path
            else:
                self.logger.warning(f"Parquet文件不存在: {parquet_path}")
                return None
        except Exception as e:
            self.logger.error(f"获取Parquet路径失败: episode={episode_id}, error={e}")
            return None

    def detect_static_frames(self, episode_id: int, video_key: str,
                           start_frame: int = 0, end_frame: Optional[int] = None,
                           threshold: Optional[float] = None, method: Optional[str] = None,
                           min_static_frames: Optional[int] = None) -> List[MotionDetectionResult]:
        """检测静止帧"""
        try:
            # 构建缓存键
            cache_key = f"{episode_id}_{video_key}_{start_frame}_{end_frame}_{threshold}_{method}_{min_static_frames}"

            # 检查缓存
            if cache_key in self._motion_cache:
                self.logger.info(f"使用缓存的运动检测结果: {cache_key}")
                return self._motion_cache[cache_key]

            # 获取视频文件路径
            video_file = self._get_video_file_path(episode_id, video_key)
            if not video_file or not video_file.exists():
                self.logger.warning(f"视频文件不存在: episode={episode_id}, video_key={video_key}")
                return []

            # 创建检测配置
            config = MotionDetectionConfig(
                method=MotionDetectionMethod.FRAME_DIFF,
                threshold=threshold or self.motion_config.threshold,
                min_static_frames=min_static_frames or self.motion_config.min_static_frames,
                resize_width=self.motion_config.resize_width,
                resize_height=self.motion_config.resize_height,
                gaussian_blur_kernel=self.motion_config.gaussian_blur_kernel
            )

            # 执行检测
            analyzer = VideoMotionAnalyzer(config)
            results = analyzer.analyze_video(video_file, start_frame, end_frame)

            # 缓存结果
            self._motion_cache[cache_key] = results

            self.logger.info(f"静止帧检测完成: episode={episode_id}, video_key={video_key}, "
                           f"总帧数={len(results)}, 静止帧数={sum(1 for r in results if r.is_static)}")

            return results

        except Exception as e:
            self.logger.error(f"静止帧检测失败: episode={episode_id}, video_key={video_key}, error={e}")
            return []

    def get_static_segments(self, episode_id: int, video_key: str,
                          threshold: Optional[float] = None, method: Optional[str] = None,
                          min_static_frames: Optional[int] = None):
        """获取静止片段"""
        try:
            # 先检测静止帧
            results = self.detect_static_frames(
                episode_id, video_key, 0, None, threshold, method, min_static_frames
            )

            if not results:
                return []

            # 创建分析器来查找静止片段
            config = MotionDetectionConfig(
                threshold=threshold or self.motion_config.threshold,
                min_static_frames=min_static_frames or self.motion_config.min_static_frames
            )
            analyzer = VideoMotionAnalyzer(config)
            segments = analyzer.find_static_segments(results)

            self.logger.info(f"获取静止片段完成: episode={episode_id}, video_key={video_key}, 找到{len(segments)}个片段")
            return segments

        except Exception as e:
            self.logger.error(f"获取静止片段失败: episode={episode_id}, video_key={video_key}, error={e}")
            return []

    def get_motion_statistics(self, episode_id: int, video_key: str,
                            threshold: Optional[float] = None, method: Optional[str] = None,
                            min_static_frames: Optional[int] = None):
        """获取运动统计信息"""
        try:
            # 先检测静止帧
            results = self.detect_static_frames(
                episode_id, video_key, 0, None, threshold, method, min_static_frames
            )

            if not results:
                return {}

            # 创建分析器来计算统计
            config = MotionDetectionConfig(
                threshold=threshold or self.motion_config.threshold,
                min_static_frames=min_static_frames or self.motion_config.min_static_frames
            )
            analyzer = VideoMotionAnalyzer(config)
            stats = analyzer.get_motion_statistics(results)

            self.logger.info(f"获取运动统计完成: episode={episode_id}, video_key={video_key}")
            return stats

        except Exception as e:
            self.logger.error(f"获取运动统计失败: episode={episode_id}, video_key={video_key}, error={e}")
            return {}

    def clear_motion_cache(self):
        """清除运动检测缓存"""
        self._motion_cache.clear()
        self.logger.info("运动检测缓存已清除")

    def plan_pruning(self, episode_id: int, video_keys: Optional[List[str]] = None,
                    threshold: Optional[float] = None, min_static_frames: Optional[int] = None) -> Dict[str, Any]:
        """制定静止帧删除计划"""
        try:
            # 确定要处理的视频键
            if video_keys is None or (isinstance(video_keys, str) and video_keys == "all"):
                video_keys = self._load_video_keys()
            elif isinstance(video_keys, str):
                video_keys = [video_keys]

            # 检测各视角的静止帧
            results_by_view = {}
            for video_key in video_keys:
                results = self.detect_static_frames(
                    episode_id, video_key, 0, None, threshold, None, min_static_frames
                )
                if results:
                    results_by_view[video_key] = results

            if not results_by_view:
                return {
                    "episode_id": episode_id,
                    "video_keys": video_keys,
                    "frame_indices": [],
                    "total_frames": 0,
                    "plan_hash": ""
                }

            # 合并检测结果
            combine_policy = self.config.get('combine_policy', 'intersection')
            analyzer = VideoMotionAnalyzer(self.motion_config)
            frame_indices = analyzer.combine_detection_results(results_by_view, combine_policy)

            # 生成计划哈希
            plan_data = {
                "episode_id": episode_id,
                "video_keys": sorted(video_keys),
                "frame_indices": sorted(frame_indices),
                "threshold": threshold or self.motion_config.threshold,
                "min_static_frames": min_static_frames or self.motion_config.min_static_frames,
                "combine_policy": combine_policy
            }
            plan_hash = hashlib.md5(json.dumps(plan_data, sort_keys=True).encode()).hexdigest()[:8]

            self.logger.info(f"删除计划制定完成: episode={episode_id}, 计划删除{len(frame_indices)}帧, hash={plan_hash}")

            return {
                "episode_id": episode_id,
                "video_keys": video_keys,
                "frame_indices": frame_indices,
                "total_frames": len(frame_indices),
                "plan_hash": plan_hash,
                "combine_policy": combine_policy
            }

        except Exception as e:
            self.logger.error(f"制定删除计划失败: episode={episode_id}, error={e}")
            return {
                "episode_id": episode_id,
                "video_keys": video_keys or [],
                "frame_indices": [],
                "total_frames": 0,
                "plan_hash": "",
                "error": str(e)
            }

    def delete_frames(self, episode_id: int, frame_indices: List[int],
                     video_keys: Optional[List[str]] = None, create_backup: bool = True) -> Dict[str, Any]:
        """执行静止帧删除（五层同步删除）"""
        start_time = time.time()

        # 获取episode级锁
        lock_file = self.locks_dir / f"episode_{episode_id}.lock"

        try:
            with open(lock_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                # 删除时必须删除所有视角的对应帧，以保持视频同步
                all_video_keys = self._load_video_keys()

                # 记录原始请求的视角（用于日志和计划哈希）
                if video_keys is None or (isinstance(video_keys, str) and video_keys == "all"):
                    requested_video_keys = all_video_keys
                elif isinstance(video_keys, str):
                    requested_video_keys = [video_keys]
                else:
                    requested_video_keys = video_keys

                # 生成计划哈希用于幂等性检查（使用请求的视角）
                plan_data = {
                    "episode_id": episode_id,
                    "video_keys": sorted(requested_video_keys),
                    "frame_indices": sorted(frame_indices)
                }
                plan_hash = hashlib.md5(json.dumps(plan_data, sort_keys=True).encode()).hexdigest()[:8]

                self.logger.info(f"开始执行静止帧删除: episode={episode_id}, 删除{len(frame_indices)}帧, hash={plan_hash}")

                # 检查是否需要创建备份
                backup_created = False
                if create_backup:
                    backup_created = self._ensure_backup()

                # 执行五层删除（删除所有视角的对应帧）
                result = self._perform_five_layer_deletion(episode_id, frame_indices, all_video_keys, plan_hash)

                # 注释掉记录操作快照的功能，避免生成log文件
                # self._record_operation_snapshot(episode_id, plan_hash, result, start_time)

                # 添加备份信息到结果
                if backup_created:
                    result["backup_created"] = True

                processing_time = time.time() - start_time
                result["processing_time"] = processing_time

                self.logger.info(f"静止帧删除完成: episode={episode_id}, 耗时={processing_time:.2f}s")

                return result

        except BlockingIOError:
            return {
                "success": False,
                "episode_id": episode_id,
                "error": "Episode正在被其他操作锁定，请稍后重试"
            }
        except Exception as e:
            self.logger.error(f"静止帧删除失败: episode={episode_id}, error={e}")
            return {
                "success": False,
                "episode_id": episode_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        finally:
            # 清理锁文件
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except:
                    pass

    def _ensure_backup(self) -> bool:
        """确保备份数据集存在，如果不存在则创建"""
        try:
            # 检查是否已存在备份
            backup_path = self.dataset_path.parent / f"{self.dataset_path.name}_backup"
            if backup_path.exists():
                self.logger.info(f"备份数据集已存在，跳过创建: {backup_path}")
                return False

            # 创建备份
            self.logger.info(f"创建备份数据集: {backup_path}")
            shutil.copytree(self.dataset_path, backup_path)
            self.logger.info("备份数据集创建完成")
            return True

        except Exception as e:
            self.logger.error(f"创建备份失败: {e}")
            raise Exception(f"无法创建备份数据集: {e}")

    def _perform_five_layer_deletion(self, episode_id: int, frame_indices: List[int],
                                   video_keys: List[str], plan_hash: str) -> Dict[str, Any]:
        """执行五层删除操作"""
        try:
            if not frame_indices:
                return {
                    "success": True,
                    "episode_id": episode_id,
                    "requested": 0,
                    "deleted": 0,
                    "failed_indices": [],
                    "per_view": {}
                }

            # 1. 视频层删除
            video_result = self._delete_video_frames(episode_id, frame_indices, video_keys)

            # 2. Parquet数据删除与重建
            parquet_result = self._delete_parquet_rows(episode_id, frame_indices)

            # 3. 元数据同步更新
            metadata_result = self._update_metadata(episode_id, len(frame_indices), video_keys)

            # 4. 一致性校验
            consistency_result = self._verify_consistency(episode_id)

            # 汇总结果
            success = (video_result.get("success", False) and
                      parquet_result.get("success", False) and
                      metadata_result.get("success", False) and
                      consistency_result.get("success", False))

            result = {
                "success": success,
                "episode_id": episode_id,
                "requested": len(frame_indices),
                "deleted": parquet_result.get("deleted", 0),
                "failed_indices": video_result.get("failed_indices", []),
                "per_view": video_result.get("per_view", {}),
                "plan_hash": plan_hash
            }

            if not success:
                errors = []
                if not video_result.get("success"):
                    errors.append(f"视频删除失败: {video_result.get('error', '')}")
                if not parquet_result.get("success"):
                    errors.append(f"数据删除失败: {parquet_result.get('error', '')}")
                if not metadata_result.get("success"):
                    errors.append(f"元数据更新失败: {metadata_result.get('error', '')}")
                if not consistency_result.get("success"):
                    errors.append(f"一致性校验失败: {consistency_result.get('error', '')}")
                result["error"] = "; ".join(errors)

            return result

        except Exception as e:
            self.logger.error(f"五层删除操作失败: episode={episode_id}, error={e}")
            return {
                "success": False,
                "episode_id": episode_id,
                "requested": len(frame_indices),
                "deleted": 0,
                "failed_indices": frame_indices,
                "per_view": {},
                "error": str(e)
            }

    def _delete_video_frames(self, episode_id: int, frame_indices: List[int],
                           video_keys: List[str]) -> Dict[str, Any]:
        """删除视频帧（所有视角）- 使用ffmpeg确保视频质量"""
        try:
            per_view = {}
            failed_indices = []
            total_deleted = 0

            for video_key in video_keys:
                video_path = self._get_video_file_path(episode_id, video_key)
                if not video_path or not video_path.exists():
                    self.logger.warning(f"视频文件不存在，跳过: {video_path}")
                    per_view[video_key] = {"deleted": 0, "error": "文件不存在"}
                    continue

                try:
                    # 使用FFmpeg重写视频，删除指定帧
                    deleted_count = self._delete_video_frames_with_ffmpeg(video_path, frame_indices)
                    per_view[video_key] = {"deleted": deleted_count}
                    total_deleted += deleted_count

                    self.logger.info(f"视频帧删除完成: {video_key}, 删除{deleted_count}帧")

                except Exception as e:
                    self.logger.error(f"视频帧删除失败: {video_key}, error={e}")
                    per_view[video_key] = {"deleted": 0, "error": str(e)}
                    failed_indices.extend(frame_indices)

            success = len([v for v in per_view.values() if "error" not in v]) > 0

            return {
                "success": success,
                "per_view": per_view,
                "failed_indices": list(set(failed_indices)),
                "total_deleted": total_deleted
            }

        except Exception as e:
            self.logger.error(f"视频层删除失败: episode={episode_id}, error={e}")
            return {
                "success": False,
                "per_view": {},
                "failed_indices": frame_indices,
                "total_deleted": 0,
                "error": str(e)
            }

    def _delete_video_frames_with_ffmpeg(self, video_path: Path, frame_indices: List[int]) -> int:
        """使用FFmpeg删除视频帧，参考duxinyu分支的实现"""
        try:
            # 获取视频信息
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            self.logger.info(f"原始视频信息: {total_frames}帧, {fps}fps")

            if not frame_indices:
                return 0

            # 计算预期结果
            new_frame_count = total_frames - len(frame_indices)
            expected_duration = new_frame_count / fps

            self.logger.info(f"预期结果: {new_frame_count}帧, {fps}fps, {expected_duration:.3f}秒")

            # 使用FFmpeg删除多个帧
            temp_path = video_path.with_suffix('.tmp.mp4')

            # 构建select表达式，删除指定的帧
            # 使用not(eq(n,frame1)+eq(n,frame2)+...)的形式
            delete_expr = "+".join([f"eq(n,{i})" for i in frame_indices])
            select_expr = f"not({delete_expr})"

            # 构建FFmpeg命令 - 参考duxinyu分支的实现
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
                '-r', str(int(fps)),  # 明确设置输出帧率
                '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
                str(temp_path), '-y'
            ]

            self.logger.info(f"执行FFmpeg命令，删除{len(frame_indices)}帧")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # 验证输出视频
                verify_cap = cv2.VideoCapture(str(temp_path))
                if verify_cap.isOpened():
                    actual_frame_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    actual_fps = verify_cap.get(cv2.CAP_PROP_FPS)
                    actual_duration = actual_frame_count / actual_fps if actual_fps > 0 else 0
                    verify_cap.release()

                    self.logger.info(f"实际结果: {actual_frame_count}帧, {actual_fps}fps, {actual_duration:.3f}秒")

                    if actual_frame_count != new_frame_count:
                        self.logger.warning(f"帧数不匹配: 期望{new_frame_count}, 实际{actual_frame_count}")
                else:
                    self.logger.warning("无法验证输出视频")

                # 原子替换
                temp_path.replace(video_path)

                deleted_count = total_frames - actual_frame_count
                return deleted_count
            else:
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()

                raise RuntimeError(f"FFmpeg处理失败: {result.stderr}")

        except Exception as e:
            self.logger.error(f"视频处理失败: {e}")
            raise

    def _delete_parquet_rows(self, episode_id: int, frame_indices: List[int]) -> Dict[str, Any]:
        """删除Parquet行并重建关键字段"""
        try:
            parquet_path = self._get_parquet_file_path(episode_id)
            if not parquet_path or not parquet_path.exists():
                return {"success": False, "error": "Parquet文件不存在", "deleted": 0}

            # 读取Parquet文件
            df = pd.read_parquet(parquet_path)
            original_count = len(df)

            # 过滤要删除的帧
            frames_to_drop = set(frame_indices)
            df_filtered = df[~df['frame_index'].isin(frames_to_drop)].copy()

            # 重建关键字段
            df_filtered['frame_index'] = np.arange(len(df_filtered))

            # 重建index字段（保持全局连续性）
            if 'index' in df_filtered.columns and len(df_filtered) > 0:
                global_index_start = df.iloc[0]['index'] if 'index' in df.columns else 0
                df_filtered['index'] = np.arange(global_index_start, global_index_start + len(df_filtered))

            # 重建timestamp字段
            if 'timestamp' in df_filtered.columns:
                fps = self._load_fps()
                df_filtered['timestamp'] = df_filtered['frame_index'] / fps

            # 写回临时文件然后原子替换
            temp_path = parquet_path.with_suffix('.tmp.parquet')
            df_filtered.to_parquet(temp_path, index=False)
            temp_path.replace(parquet_path)

            deleted_count = original_count - len(df_filtered)

            self.logger.info(f"Parquet行删除完成: episode={episode_id}, 删除{deleted_count}行")

            return {
                "success": True,
                "deleted": deleted_count,
                "remaining": len(df_filtered)
            }

        except Exception as e:
            self.logger.error(f"Parquet行删除失败: episode={episode_id}, error={e}")
            return {"success": False, "error": str(e), "deleted": 0}

    def _update_metadata(self, episode_id: int, deleted_frames: int, video_keys: List[str]) -> Dict[str, Any]:
        """更新元数据文件"""
        try:
            # 更新episodes.jsonl
            episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
            episodes = load_json_lines(episodes_path)

            for episode in episodes:
                if episode.get('episode_index') == episode_id:
                    # 获取更新后的帧数
                    parquet_path = self._get_parquet_file_path(episode_id)
                    if parquet_path and parquet_path.exists():
                        df = pd.read_parquet(parquet_path)
                        new_length = len(df)
                        episode['length'] = new_length

                        # 不修改其他字段，只更新length
                    break

            # 写回episodes.jsonl
            temp_episodes_path = episodes_path.with_suffix('.tmp.jsonl')
            save_json_lines(episodes, temp_episodes_path)
            temp_episodes_path.replace(episodes_path)

            # 重新计算episodes_stats.jsonl
            self._update_episode_stats(episode_id)

            # 更新info.json
            info_path = self.dataset_path / "meta" / "info.json"
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)

            # 重新计算total_frames（所有视角视频的总帧数）
            total_frames = 0
            total_videos = 0

            for episode in episodes:
                episode_length = episode.get('length', 0)
                episode_id_curr = episode.get('episode_index', 0)

                # 计算该episode实际存在的视角数（使用所有可能的视角）
                all_video_keys = self._load_video_keys()
                existing_views = 0
                for video_key in all_video_keys:
                    video_path = self._get_video_file_path(episode_id_curr, video_key)
                    if video_path and video_path.exists():
                        existing_views += 1

                total_frames += episode_length * existing_views
                total_videos += existing_views

            info['total_frames'] = total_frames
            info['total_videos'] = total_videos

            # 写回info.json
            temp_info_path = info_path.with_suffix('.tmp.json')
            with open(temp_info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            temp_info_path.replace(info_path)

            self.logger.info(f"元数据更新完成: episode={episode_id}")

            return {"success": True}

        except Exception as e:
            self.logger.error(f"元数据更新失败: episode={episode_id}, error={e}")
            return {"success": False, "error": str(e)}

    def _verify_consistency(self, episode_id: int) -> Dict[str, Any]:
        """一致性校验"""
        try:
            # 检查Parquet与episodes.jsonl的一致性
            parquet_path = self._get_parquet_file_path(episode_id)
            if not parquet_path or not parquet_path.exists():
                return {"success": False, "error": "Parquet文件不存在"}

            df = pd.read_parquet(parquet_path)
            parquet_length = len(df)

            episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
            episodes = load_json_lines(episodes_path)

            episode_length = None
            for episode in episodes:
                if episode.get('episode_index') == episode_id:
                    episode_length = episode.get('length', 0)
                    break

            if episode_length is None:
                return {"success": False, "error": "在episodes.jsonl中未找到对应episode"}

            if parquet_length != episode_length:
                return {
                    "success": False,
                    "error": f"长度不一致: parquet={parquet_length}, episodes.jsonl={episode_length}"
                }

            self.logger.info(f"一致性校验通过: episode={episode_id}")
            return {"success": True}

        except Exception as e:
            self.logger.error(f"一致性校验失败: episode={episode_id}, error={e}")
            return {"success": False, "error": str(e)}

    def _record_operation_snapshot(self, episode_id: int, plan_hash: str,
                                 result: Dict[str, Any], start_time: float):
        """记录操作快照"""
        try:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "episode_id": episode_id,
                "plan_hash": plan_hash,
                "operation": "static_frame_deletion",
                "result": result,
                "processing_time": time.time() - start_time
            }

            snapshot_file = self.ops_logs_dir / f"static_prune_{episode_id}_{int(start_time)}.json"
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            self.logger.info(f"操作快照已记录: {snapshot_file}")

        except Exception as e:
            self.logger.warning(f"记录操作快照失败: {e}")

    def _update_episode_stats(self, episode_id: int):
        """重新计算指定episode的统计数据"""
        try:
            # 读取更新后的Parquet数据
            parquet_path = self._get_parquet_file_path(episode_id)
            if not parquet_path or not parquet_path.exists():
                self.logger.warning(f"无法更新统计数据，Parquet文件不存在: episode={episode_id}")
                return

            df = pd.read_parquet(parquet_path)

            # 读取episodes_stats.jsonl
            stats_path = self.dataset_path / "meta" / "episodes_stats.jsonl"
            if not stats_path.exists():
                self.logger.warning(f"episodes_stats.jsonl不存在，跳过统计更新")
                return

            stats_data = load_json_lines(stats_path)

            # 找到对应episode的统计数据并更新
            for stats_entry in stats_data:
                if stats_entry.get('episode_index') == episode_id:
                    # 重新计算统计数据
                    new_stats = self._compute_episode_stats(df, episode_id)
                    stats_entry['stats'] = new_stats
                    break

            # 写回episodes_stats.jsonl
            temp_stats_path = stats_path.with_suffix('.tmp.jsonl')
            save_json_lines(stats_data, temp_stats_path)
            temp_stats_path.replace(stats_path)

            self.logger.info(f"Episode统计数据更新完成: episode={episode_id}")

        except Exception as e:
            self.logger.error(f"更新episode统计数据失败: episode={episode_id}, error={e}")

    def _compute_episode_stats(self, df: pd.DataFrame, episode_id: Optional[int] = None) -> Dict[str, Any]:
        """计算单个episode的统计数据"""
        stats = {}

        # 先处理DataFrame中存在的列
        for column in df.columns:
            try:
                if column in ['observation.images.laptop', 'observation.images.phone']:
                    # DataFrame中有图像列（通常不会出现，但为完整性保留处理）
                    if episode_id is not None:
                        # 从列名提取视频键
                        video_key = column.split('.')[-1]  # 获取laptop或phone
                        # 构建视频文件路径
                        video_path = self._get_video_file_path(episode_id, video_key)
                        if video_path and video_path.exists():
                            # 从视频计算真实的图像统计
                            episode_length = len(df)
                            image_stats = compute_image_stats_from_video(video_path, episode_length)
                            # 确保统计结果格式正确
                            stats[column] = {
                                "min": image_stats["min"].tolist() if isinstance(image_stats["min"], np.ndarray) else image_stats["min"],
                                "max": image_stats["max"].tolist() if isinstance(image_stats["max"], np.ndarray) else image_stats["max"],
                                "mean": image_stats["mean"].tolist() if isinstance(image_stats["mean"], np.ndarray) else image_stats["mean"],
                                "std": image_stats["std"].tolist() if isinstance(image_stats["std"], np.ndarray) else image_stats["std"],
                                "count": image_stats["count"].tolist() if isinstance(image_stats["count"], np.ndarray) else image_stats["count"]
                            }
                        else:
                            # 视频文件不存在，使用默认值
                            stats[column] = {
                                "min": [[[0.0]], [[0.0]], [[0.0]]],
                                "max": [[[1.0]], [[1.0]], [[1.0]]],
                                "mean": [[[0.5]], [[0.5]], [[0.5]]],
                                "std": [[[0.3]], [[0.3]], [[0.3]]],
                                "count": [100]
                            }
                    else:
                        # 没有episode_id，使用默认值
                        stats[column] = {
                            "min": [[[0.0]], [[0.0]], [[0.0]]],
                            "max": [[[1.0]], [[1.0]], [[1.0]]],
                            "mean": [[[0.5]], [[0.5]], [[0.5]]],
                            "std": [[[0.3]], [[0.3]], [[0.3]]],
                            "count": [100]
                        }
                else:
                    # 数值列的统计
                    col_data = df[column]
                    if col_data.dtype in ['int64', 'float64']:
                        if len(col_data) > 0:
                            stats[column] = {
                                "min": [float(col_data.min())],
                                "max": [float(col_data.max())],
                                "mean": [float(col_data.mean())],
                                "std": [float(col_data.std())],
                                "count": [len(col_data)]
                            }
                    elif hasattr(col_data.iloc[0], '__len__') and not isinstance(col_data.iloc[0], str):
                        # 数组类型的列（如action）
                        try:
                            # 转换为numpy数组进行统计
                            array_data = np.array(col_data.tolist())
                            if array_data.ndim == 2:  # 二维数组
                                stats[column] = {
                                    "min": array_data.min(axis=0).tolist(),
                                    "max": array_data.max(axis=0).tolist(),
                                    "mean": array_data.mean(axis=0).tolist(),
                                    "std": array_data.std(axis=0).tolist(),
                                    "count": [len(array_data)]
                                }
                            else:
                                # 一维数组
                                stats[column] = {
                                    "min": [float(array_data.min())],
                                    "max": [float(array_data.max())],
                                    "mean": [float(array_data.mean())],
                                    "std": [float(array_data.std())],
                                    "count": [len(array_data)]
                                }
                        except Exception as e:
                            self.logger.warning(f"计算数组列统计失败: {column}, error={e}")
                            stats[column] = {"count": [len(col_data)]}
                    else:
                        # 其他类型，只记录count
                        stats[column] = {"count": [len(col_data)]}

            except Exception as e:
                self.logger.warning(f"计算列统计失败: {column}, error={e}")
                stats[column] = {"count": [len(df)]}

        # 检查并添加可能存在的图像视频统计（即使DataFrame中没有图像列）
        if episode_id is not None:
            for video_key in ['laptop', 'phone']:
                image_column = f'observation.images.{video_key}'

                # 如果DataFrame中没有这个图像列，但视频文件存在，则计算统计
                if image_column not in stats:
                    video_path = self._get_video_file_path(episode_id, video_key)
                    if video_path and video_path.exists():
                        try:
                            episode_length = len(df)
                            image_stats = compute_image_stats_from_video(video_path, episode_length)

                            # 添加图像统计到结果中
                            stats[image_column] = {
                                "min": image_stats["min"].tolist() if isinstance(image_stats["min"], np.ndarray) else image_stats["min"],
                                "max": image_stats["max"].tolist() if isinstance(image_stats["max"], np.ndarray) else image_stats["max"],
                                "mean": image_stats["mean"].tolist() if isinstance(image_stats["mean"], np.ndarray) else image_stats["mean"],
                                "std": image_stats["std"].tolist() if isinstance(image_stats["std"], np.ndarray) else image_stats["std"],
                                "count": image_stats["count"].tolist() if isinstance(image_stats["count"], np.ndarray) else image_stats["count"]
                            }

                            self.logger.info(f"从视频文件计算图像统计: {image_column}")
                        except Exception as e:
                            self.logger.warning(f"从视频计算图像统计失败: {image_column}, error={e}")

        return stats