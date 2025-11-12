#!/usr/bin/env python3
"""
基于动作序列聚合的LeRobot数据重构脚本
功能: 根据frames.jsonl中的标注信息，将不同episode中相同技能的片段聚合成独立的数据集
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.config import DatasetConfig


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


class SkillBasedSplitter:
    """基于技能的数据分割器"""
    
    def __init__(self, dataset_root: str, frames_jsonl: str = None, out_root: str = None):
        self.dataset_config = DatasetConfig(dataset_root)

        # 设置默认的 frames.jsonl 路径
        if frames_jsonl is None:
            frames_jsonl = "/home/chenqingyu/robot/lerobot_data/datasets/grasp_dataset/meta/frames.jsonl"
        self.frames_jsonl = Path(frames_jsonl)

        # 设置默认的输出路径
        if out_root is None:
            out_root = "/home/chenqingyu/robot/lerobot_data/output"
        self.out_root = Path(out_root)

        self.fps = 30  # 从文档中获取的FPS

        # 技能到任务索引的映射 - 将动态生成
        self.skill_to_task_index = {}

        # 缓存解析结果
        self._cached_segments = None
        self._cached_skills = None

        # 设置日志
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)
    
    def extract_all_skills_and_segments(self) -> Tuple[List[EpisodeSegment], List[str]]:
        """解析frames.jsonl文件，同时提取所有片段信息和技能列表"""
        if self._cached_segments is not None and self._cached_skills is not None:
            return self._cached_segments, self._cached_skills

        segments = []
        skills_set = set()

        if not self.frames_jsonl.exists():
            raise FileNotFoundError(f"frames.jsonl文件不存在: {self.frames_jsonl}")

        # 尝试两种格式：JSONL（每行一个JSON）和JSON数组
        try:
            with open(self.frames_jsonl, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                # 检查是否是JSON数组格式
                if content.startswith('['):
                    self.logger.info("检测到JSON数组格式，解析中...")
                    data_list = json.loads(content)

                    for data in data_list:
                        episode_id = data['episode_id']

                        for action in data['label_info']['action_config']:
                            start_frame = action['start_frame']
                            end_frame = action['end_frame']
                            skill = action['skill']
                            action_text = action['action_text']
                            length = end_frame - start_frame + 1

                            # 收集技能
                            skills_set.add(skill)

                            segment = EpisodeSegment(
                                episode_id=episode_id,
                                skill=skill,
                                start_frame=start_frame,
                                end_frame=end_frame,
                                action_text=action_text,
                                length=length
                            )
                            segments.append(segment)
                else:
                    # JSONL格式（每行一个JSON对象）
                    self.logger.info("检测到JSONL格式，解析中...")
                    for line_num, line in enumerate(content.split('\n'), 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            episode_id = data['episode_id']

                            for action in data['label_info']['action_config']:
                                start_frame = action['start_frame']
                                end_frame = action['end_frame']
                                skill = action['skill']
                                action_text = action['action_text']
                                length = end_frame - start_frame + 1

                                # 收集技能
                                skills_set.add(skill)

                                segment = EpisodeSegment(
                                    episode_id=episode_id,
                                    skill=skill,
                                    start_frame=start_frame,
                                    end_frame=end_frame,
                                    action_text=action_text,
                                    length=length
                                )
                                segments.append(segment)

                        except (json.JSONDecodeError, KeyError) as e:
                            self.logger.error(f"解析frames.jsonl第{line_num}行失败: {e}")
                            continue

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"解析frames.jsonl文件失败: {e}")
            raise

        # 按字母顺序排序技能列表
        skills_list = sorted(list(skills_set))

        # 生成技能到任务索引的映射
        self.skill_to_task_index = {skill: idx for idx, skill in enumerate(skills_list)}

        # 缓存结果
        self._cached_segments = segments
        self._cached_skills = skills_list

        self.logger.info(f"解析完成，共找到 {len(segments)} 个片段")
        self.logger.info(f"发现 {len(skills_list)} 种技能: {skills_list}")
        self.logger.info(f"技能映射: {self.skill_to_task_index}")

        return segments, skills_list

    def parse_markers(self) -> List[EpisodeSegment]:
        """解析frames.jsonl文件，提取所有片段信息（保持向后兼容）"""
        segments, _ = self.extract_all_skills_and_segments()
        return segments
    
    def iter_segments_by_skill(self, segments: List[EpisodeSegment], skill: str) -> List[EpisodeSegment]:
        """按技能过滤片段，并按episode_id排序"""
        skill_segments = [seg for seg in segments if seg.skill == skill]
        skill_segments.sort(key=lambda x: x.episode_id)
        
        self.logger.info(f"技能 {skill} 共有 {len(skill_segments)} 个片段")
        return skill_segments
    
    def cut_parquet(self, src_parquet: Path, start_frame: int, end_frame: int,
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
        
        # 更新task_index
        task_index = self.skill_to_task_index[skill]
        new_data['task_index'] = pa.array([task_index] * len(filtered_table), type=pa.int64())
        
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
    
    def write_parquet(self, table: pa.Table, out_path: Path):
        """写入Parquet文件"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        self.logger.debug(f"Parquet文件已保存: {out_path}")
    
    def cut_video_ffmpeg(self, src_mp4: Path, t_start: float, t_end_exclusive: float, 
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
    
    def get_video_duration(self, video_file: Path) -> float:
        """获取视频时长"""
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            self.logger.error(f"获取视频时长失败: {e}")
            raise
    
    def get_video_frame_count(self, video_file: Path) -> int:
        """获取视频帧数"""
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_frames",
            "-of", "csv=p=0", str(video_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return int(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            self.logger.error(f"获取视频帧数失败: {e}")
            raise

    def emit_meta(self, out_root: Path, episodes_meta: List[Dict], skill: str):
        """生成元数据文件"""
        meta_dir = out_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # 生成tasks.jsonl
        tasks_file = meta_dir / "tasks.jsonl"
        with open(tasks_file, 'w', encoding='utf-8') as f:
            for skill_name, task_index in self.skill_to_task_index.items():
                task_data = {"task_index": task_index, "task": skill_name}
                f.write(json.dumps(task_data) + '\n')

        # 生成episodes.jsonl
        episodes_file = meta_dir / "episodes.jsonl"
        with open(episodes_file, 'w', encoding='utf-8') as f:
            for episode_meta in episodes_meta:
                f.write(json.dumps(episode_meta) + '\n')

        # 生成info.json
        self._generate_info_json(out_root, episodes_meta, skill)

        # 生成episodes_stats.jsonl (简化版本)
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
        """生成episodes_stats.jsonl文件（遵循LeRobot原始计算逻辑）"""
        import numpy as np

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
        """计算单个episode的统计信息，严格遵循LeRobot原始计算逻辑和字段顺序"""
        import numpy as np
        from collections import OrderedDict

        # 读取Parquet数据
        table = pq.read_table(parquet_file)
        episode_length = len(table)

        # 使用OrderedDict保持字段顺序，严格按照原始数据格式
        stats = OrderedDict()

        # 1. action字段 (第一个)
        if 'action' in table.column_names:
            arrays = np.stack(table['action'].to_pylist())
            stats['action'] = {
                'min': np.min(arrays, axis=0).tolist(),
                'max': np.max(arrays, axis=0).tolist(),
                'mean': np.mean(arrays, axis=0).tolist(),
                'std': np.std(arrays, axis=0).tolist(),
                'count': [episode_length]
            }

        # 2. observation.state字段 (第二个)
        if 'observation.state' in table.column_names:
            arrays = np.stack(table['observation.state'].to_pylist())
            stats['observation.state'] = {
                'min': np.min(arrays, axis=0).tolist(),
                'max': np.max(arrays, axis=0).tolist(),
                'mean': np.mean(arrays, axis=0).tolist(),
                'std': np.std(arrays, axis=0).tolist(),
                'count': [episode_length]
            }

        # 3. 图像字段 (第三、四个) - 保持原始的采样统计信息（count=100）
        image_fields = ['observation.images.laptop', 'observation.images.phone']
        for field in image_fields:
            original_stats = self._get_image_field_stats(field)
            if original_stats:
                stats[field] = original_stats

        # 4. 标量字段 (按原始顺序：timestamp, frame_index, episode_index, index, task_index)
        scalar_fields = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
        for field in scalar_fields:
            if field in table.column_names:
                values = table[field].to_numpy()

                # 处理不同数据类型
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
                    # 浮点数字段 (timestamp)
                    stats[field] = {
                        'min': [float(np.min(values))],
                        'max': [float(np.max(values))],
                        'mean': [float(np.mean(values))],
                        'std': [float(np.std(values))],
                        'count': [episode_length]
                    }

        return dict(stats)  # 转换为普通dict

    def _get_image_field_stats(self, field: str) -> Dict[str, Any]:
        """获取图像字段的统计信息（从原始数据集中采样）"""
        # 从原始数据集中读取图像统计信息作为模板
        try:
            original_stats_file = self.dataset_config.dataset_path / "meta" / "episodes_stats.jsonl"
            if original_stats_file.exists():
                with open(original_stats_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        original_stats = json.loads(first_line)
                        if field in original_stats['stats']:
                            # 返回原始的图像统计信息，保持count=100
                            return original_stats['stats'][field]
        except Exception as e:
            self.logger.warning(f"无法获取原始图像统计信息: {e}")

        # 如果无法获取原始统计信息，返回None
        return None

    def validate_episode_pair(self, out_root: Path, episode_index: int) -> ValidationReport:
        """验证单个episode的数据一致性"""
        errors = []
        warnings = []

        try:
            # 检查Parquet文件
            parquet_file = out_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            if not parquet_file.exists():
                errors.append(f"Parquet文件不存在: {parquet_file}")
                return ValidationReport(False, errors, warnings)

            table = pq.read_table(parquet_file)
            parquet_frames = len(table)

            # 检查视频文件
            video_keys = ["observation.images.laptop", "observation.images.phone"]
            video_frame_counts = []

            for video_key in video_keys:
                video_file = out_root / "videos" / "chunk-000" / video_key / f"episode_{episode_index:06d}.mp4"
                if not video_file.exists():
                    errors.append(f"视频文件不存在: {video_file}")
                    continue

                try:
                    frame_count = self.get_video_frame_count(video_file)
                    video_frame_counts.append(frame_count)
                except Exception as e:
                    warnings.append(f"无法获取视频帧数 {video_file}: {e}")

            # 验证帧数一致性
            if video_frame_counts:
                # 检查Parquet与视频帧数一致性（±1帧容差）
                for i, video_frames in enumerate(video_frame_counts):
                    if abs(parquet_frames - video_frames) > 1:
                        warnings.append(f"帧数不一致: Parquet={parquet_frames}, Video{i}={video_frames}")

                # 检查双路视频帧数一致性
                if len(video_frame_counts) == 2:
                    if abs(video_frame_counts[0] - video_frame_counts[1]) > 1:
                        warnings.append(f"双路视频帧数不一致: laptop={video_frame_counts[0]}, phone={video_frame_counts[1]}")

            # 验证字段正确性
            if 'frame_index' in table.column_names:
                frame_indices = table['frame_index'].to_pylist()
                if frame_indices != list(range(len(frame_indices))):
                    errors.append("frame_index字段不是从0开始的连续序列")

            if 'timestamp' in table.column_names:
                timestamps = table['timestamp'].to_pylist()
                expected_timestamps = [i / self.fps for i in range(len(timestamps))]
                if not all(abs(a - b) < 0.001 for a, b in zip(timestamps, expected_timestamps)):
                    warnings.append("timestamp字段与期望值不匹配")

        except Exception as e:
            errors.append(f"验证过程中发生异常: {e}")

        success = len(errors) == 0
        return ValidationReport(success, errors, warnings)

    def validate_snapshot(self, out_root: Path) -> ValidationReport:
        """验证整个snapshot的一致性"""
        all_errors = []
        all_warnings = []

        # 检查meta文件
        meta_files = ["info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]
        for meta_file in meta_files:
            meta_path = out_root / "meta" / meta_file
            if not meta_path.exists():
                all_errors.append(f"元数据文件不存在: {meta_path}")

        # 读取episodes.jsonl获取episode列表
        episodes_file = out_root / "meta" / "episodes.jsonl"
        if episodes_file.exists():
            try:
                with open(episodes_file, 'r', encoding='utf-8') as f:
                    episodes = []
                    for line in f:
                        if line.strip():
                            episodes.append(json.loads(line.strip()))

                # 验证每个episode
                for episode_meta in episodes:
                    episode_index = episode_meta['episode_index']
                    report = self.validate_episode_pair(out_root, episode_index)
                    all_errors.extend(report.errors)
                    all_warnings.extend(report.warnings)

            except Exception as e:
                all_errors.append(f"读取episodes.jsonl失败: {e}")

        success = len(all_errors) == 0
        return ValidationReport(success, all_errors, all_warnings)

    def make_out_root(self, skill: str) -> Path:
        """创建输出根目录"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = self.out_root / f"koch__skill-{skill.lower()}-{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def process_skill(self, skill: str, segments: List[EpisodeSegment],
                     reindex: bool = True, reencode_on_gop: bool = False,
                     strict_validation: bool = True, dry_run: bool = False):
        """处理单个技能的所有片段"""
        self.logger.info(f"开始处理技能: {skill}")

        # 获取该技能的所有片段
        skill_segments = self.iter_segments_by_skill(segments, skill)
        if not skill_segments:
            self.logger.warning(f"技能 {skill} 没有找到任何片段")
            return

        # 创建输出目录
        out_root = self.make_out_root(skill)
        self.logger.info(f"输出目录: {out_root}")

        if dry_run:
            self.logger.info("*** DRY RUN 模式 - 不会实际创建文件 ***")

        episodes_meta = []

        # 处理每个片段
        for new_idx, segment in enumerate(skill_segments):
            self.logger.info(f"处理片段 {new_idx + 1}/{len(skill_segments)}: "
                           f"Episode {segment.episode_id}, {segment.skill}, "
                           f"帧 {segment.start_frame}-{segment.end_frame}")

            try:
                # 处理Parquet数据
                src_parquet = self.dataset_config.get_data_path(segment.episode_id)
                if not dry_run:
                    table = self.cut_parquet(src_parquet, segment.start_frame, segment.end_frame,
                                           new_idx, skill, segment)
                    out_parquet = out_root / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
                    self.write_parquet(table, out_parquet)

                # 处理视频文件（调整end_frame逻辑）
                video_keys = ["observation.images.laptop", "observation.images.phone"]
                t_start = segment.start_frame / self.fps
                # 调整视频切割逻辑：end_frame减1后再+1，相当于直接使用原始end_frame作为exclusive边界
                t_end_exclusive = segment.end_frame / self.fps

                for video_key in video_keys:
                    src_video = self.dataset_config.get_video_path(segment.episode_id, video_key)
                    out_video = out_root / "videos" / "chunk-000" / video_key / f"episode_{new_idx:06d}.mp4"

                    if not dry_run:
                        success = self.cut_video_ffmpeg(src_video, t_start, t_end_exclusive,
                                                      out_video, reencode_on_gop)
                        if not success:
                            self.logger.error(f"视频切割失败: {src_video} -> {out_video}")
                            continue

                # 记录episode元数据 - 严格按照原始格式，只保留必要字段
                # 调整长度计算：end_frame减1后的实际长度
                actual_length = segment.length - 1
                episode_meta = {
                    "episode_index": new_idx,
                    "tasks": [skill],  # 使用技能名作为任务
                    "length": actual_length
                }
                episodes_meta.append(episode_meta)

            except Exception as e:
                self.logger.error(f"处理片段失败: {e}")
                continue

        if not dry_run:
            # 生成元数据
            self.emit_meta(out_root, episodes_meta, skill)

            # 验证结果
            if strict_validation:
                self.logger.info("开始验证结果...")
                report = self.validate_snapshot(out_root)

                if report.success:
                    self.logger.info("✓ 验证通过")
                else:
                    self.logger.error("✗ 验证失败")
                    for error in report.errors:
                        self.logger.error(f"  错误: {error}")

                if report.warnings:
                    for warning in report.warnings:
                        self.logger.warning(f"  警告: {warning}")

        self.logger.info(f"技能 {skill} 处理完成，共 {len(episodes_meta)} 个片段")

    def run(self, skills: Optional[List[str]] = None, reindex: bool = True, reencode_on_gop: bool = False,
            strict_validation: bool = True, dry_run: bool = False):
        """运行完整的分割流程"""
        self.logger.info("开始基于技能的数据分割")
        self.logger.info(f"输出根目录: {self.out_root}")

        # 解析标注文件并提取所有技能
        segments, available_skills = self.extract_all_skills_and_segments()

        # 如果没有指定技能列表，使用所有发现的技能
        if skills is None:
            skills_to_process = available_skills
            self.logger.info(f"未指定技能列表，将处理所有发现的技能: {skills_to_process}")
        else:
            skills_to_process = skills
            self.logger.info(f"指定的目标技能: {skills_to_process}")
            # 验证技能列表
            available_skills_set = set(available_skills)
            for skill in skills_to_process:
                if skill not in available_skills_set:
                    self.logger.warning(f"技能 {skill} 在标注文件中未找到，将跳过")

        # 显示每个技能的片段统计
        skill_counts = {}
        for segment in segments:
            skill_counts[segment.skill] = skill_counts.get(segment.skill, 0) + 1

        self.logger.info("技能片段统计:")
        for skill in sorted(skill_counts.keys()):
            self.logger.info(f"  {skill}: {skill_counts[skill]} 个片段")

        # 处理每个技能
        processed_skills = []
        available_skills_set = set(available_skills)
        for skill in skills_to_process:
            if skill in available_skills_set:
                try:
                    self.process_skill(skill, segments, reindex, reencode_on_gop,
                                     strict_validation, dry_run)
                    processed_skills.append(skill)
                except Exception as e:
                    self.logger.error(f"处理技能 {skill} 时发生错误: {e}")
                    continue

        self.logger.info(f"处理完成，成功处理的技能: {processed_skills}")
        if len(processed_skills) != len([s for s in skills_to_process if s in available_skills_set]):
            failed_skills = [s for s in skills_to_process if s in available_skills_set and s not in processed_skills]
            self.logger.warning(f"处理失败的技能: {failed_skills}")


def main():
    parser = argparse.ArgumentParser(description="基于动作序列聚合的LeRobot数据重构")
    parser.add_argument("--dataset-root", required=True, help="原始数据集根目录")
    parser.add_argument("--frames-jsonl", required=False, default="/home/chenqingyu/robot/lerobot_data/datasets/grasp_dataset/meta/frames.jsonl", help="frames.jsonl文件路径（默认: /home/chenqingyu/robot/lerobot_data/datasets/grasp_dataset/meta/frames.jsonl）")
    parser.add_argument("--skills", nargs='+', default=None,
                       help="要处理的技能列表；不提供则自动从frames.jsonl提取")
    parser.add_argument("--out-root", required=True, help="输出根目录")
    parser.add_argument("--reindex", action="store_true", default=True,
                       help="是否重建索引字段")
    parser.add_argument("--reencode-on-gop", action="store_true",
                       help="GOP边界问题时是否重编码")
    parser.add_argument("--strict-validation", action="store_true", default=True,
                       help="是否进行严格验证")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式")

    args = parser.parse_args()

    # 创建分割器
    splitter = SkillBasedSplitter(
        dataset_root=args.dataset_root,
        frames_jsonl=args.frames_jsonl,
        out_root=args.out_root
    )

    # 执行分割
    try:
        splitter.run(
            skills=args.skills,
            reindex=args.reindex,
            reencode_on_gop=args.reencode_on_gop,
            strict_validation=args.strict_validation,
            dry_run=args.dry_run
        )
        print("数据重构完成!")
    except Exception as e:
        print(f"数据重构失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
