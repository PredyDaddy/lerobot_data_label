"""
标注存储模块
功能: 标注数据的读写、JSONL格式处理、原子写入、数据校验
"""

import json
import fcntl
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .config import DatasetConfig


# 数据模型定义
@dataclass
class Segment:
    """标注段落数据结构"""
    start_frame: int
    end_frame: int
    skill: str
    action_text: str

    def __post_init__(self):
        """数据校验"""
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        if not self.skill or not self.skill.strip():
            raise ValueError("skill cannot be empty")
        if not self.action_text or not self.action_text.strip():
            raise ValueError("action_text cannot be empty")
        if len(self.action_text.strip()) > 500:
            raise ValueError("action_text must be at most 500 characters")


@dataclass
class EpisodeAnnotations:
    """Episode标注数据结构"""
    episode_id: int
    label_info: Dict[str, List[Dict[str, Any]]]
    task_name: Optional[str] = None
    init_scene_text: Optional[str] = None

    def __post_init__(self):
        """数据校验和格式化"""
        if self.episode_id < 0:
            raise ValueError("episode_id must be non-negative")

        # 强制验证必填字段
        if not self.task_name or not self.task_name.strip():
            raise ValueError("task_name cannot be empty")
        if not self.init_scene_text or not self.init_scene_text.strip():
            raise ValueError("init_scene_text cannot be empty")

        # 确保label_info具有正确的结构
        if "action_config" not in self.label_info:
            self.label_info["action_config"] = []

        # 校验至少有一个段落
        if len(self.label_info["action_config"]) == 0:
            raise ValueError("action_config must contain at least one segment")

        # 校验segments
        segments = []
        for segment_data in self.label_info["action_config"]:
            segment = Segment(**segment_data)
            segments.append(asdict(segment))

        # 检查重叠
        self._check_overlaps(segments)
        self.label_info["action_config"] = segments

    def _check_overlaps(self, segments: List[Dict[str, Any]]):
        """检查段落重叠（半开区间）"""
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i != j:
                    # 检查半开区间重叠: [a,b) 和 [c,d) 重叠当且仅当 max(a,c) < min(b,d)
                    # 端点相等不视为重叠（允许相邻段落）
                    if max(seg1["start_frame"], seg2["start_frame"]) < min(seg1["end_frame"], seg2["end_frame"]):
                        raise ValueError(f"Segments overlap: [{seg1['start_frame']},{seg1['end_frame']}) and [{seg2['start_frame']},{seg2['end_frame']})")


class AnnotationStore:
    """标注数据存储管理器"""

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.frames_path = dataset_config.get_frames_path()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        # 确保meta目录存在
        self.frames_path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> Dict[int, Dict[str, Any]]:
        """读取所有标注数据，返回以episode_id为键的字典"""
        annotations = {}

        if not self.frames_path.exists():
            return annotations

        try:
            with open(self.frames_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return annotations

            # 尝试解析为JSONL格式
            if self._is_jsonl_format(content):
                for line_num, line in enumerate(content.splitlines(), 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            episode_id = data.get("episode_id")
                            if episode_id is not None:
                                annotations[episode_id] = data
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
            else:
                # 尝试解析为JSON Array格式
                try:
                    data_list = json.loads(content)
                    if isinstance(data_list, list):
                        for data in data_list:
                            episode_id = data.get("episode_id")
                            if episode_id is not None:
                                annotations[episode_id] = data
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse frames.jsonl as JSON array: {e}")

        except (IOError, OSError) as e:
            self.logger.error(f"Failed to read frames.jsonl: {e}")

        return annotations

    def load(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """读取单个episode的标注数据"""
        all_annotations = self.load_all()
        return all_annotations.get(episode_id)

    def upsert(self, episode_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """插入或更新episode的标注数据"""
        with self._lock:
            # 数据校验
            try:
                # 构建完整的payload
                full_payload = {
                    "episode_id": episode_id,
                    "label_info": payload.get("label_info", {"action_config": []}),
                    "task_name": payload.get("task_name"),
                    "init_scene_text": payload.get("init_scene_text")
                }

                # 使用数据模型进行校验
                episode_annotations = EpisodeAnnotations(**full_payload)
                validated_payload = asdict(episode_annotations)

                # 移除None值以保持输出简洁
                if validated_payload["task_name"] is None:
                    validated_payload.pop("task_name")
                if validated_payload["init_scene_text"] is None:
                    validated_payload.pop("init_scene_text")

            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid annotation data: {e}")

            # 检查是否有改动
            existing_data = self.load(episode_id)
            if existing_data == validated_payload:
                return {
                    "ok": True,
                    "saved_at": datetime.now().timestamp(),
                    "count": len(validated_payload["label_info"]["action_config"]),
                    "written": False
                }

            # 写入文件 (upsert-to-array策略)
            try:
                self._upsert_to_array_file(validated_payload)

                return {
                    "ok": True,
                    "saved_at": datetime.now().timestamp(),
                    "count": len(validated_payload["label_info"]["action_config"]),
                    "written": True
                }

            except (IOError, OSError) as e:
                self.logger.error(f"Failed to write annotation: {e}")
                raise RuntimeError(f"Failed to save annotation: {e}")

    def _upsert_to_array_file(self, data: Dict[str, Any]):
        """使用upsert策略写入JSON数组格式"""
        current_episode_id = data["episode_id"]

        # 读取现有数据
        all_annotations = {}
        if self.frames_path.exists():
            with open(self.frames_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    if self._is_jsonl_format(content):
                        # JSONL格式
                        for line in content.splitlines():
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    if "episode_id" in item:
                                        all_annotations[item["episode_id"]] = item
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # JSON Array格式
                        try:
                            json_array = json.loads(content)
                            if isinstance(json_array, list):
                                for item in json_array:
                                    if "episode_id" in item:
                                        all_annotations[item["episode_id"]] = item
                        except json.JSONDecodeError:
                            pass

        # 更新/插入当前数据 (last-write-wins)
        all_annotations[current_episode_id] = data

        # 转换为数组格式
        data_array = list(all_annotations.values())

        # 使用临时文件+原子重命名确保写入安全
        temp_dir = self.frames_path.parent
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                       dir=temp_dir, delete=False,
                                       suffix='.tmp') as temp_file:
            temp_path = Path(temp_file.name)

            try:
                # 写入JSON数组格式
                json.dump(data_array, temp_file, ensure_ascii=False, indent=2)
                temp_file.flush()

                # 原子重命名
                temp_path.replace(self.frames_path)

            except Exception:
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                raise

    def _is_jsonl_format(self, content: str) -> bool:
        """判断文件内容是否为JSONL格式"""
        lines = content.strip().splitlines()
        if len(lines) == 0:
            return False

        # 检查第一行是否为有效JSON对象
        try:
            first_line = lines[0].strip()
            if first_line:
                parsed = json.loads(first_line)
                # 如果第一行是JSON对象（而不是数组），且多行或单行对象，则认为是JSONL
                # 如果第一行就是完整的数组（如 [{}]），则不是JSONL
                if isinstance(parsed, dict):
                    return True
                elif isinstance(parsed, list) and len(lines) == 1:
                    # 单行包含完整数组，不是JSONL格式
                    return False
        except json.JSONDecodeError:
            pass

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取标注统计信息"""
        all_annotations = self.load_all()

        total_episodes = len(all_annotations)
        total_segments = 0
        skills_count = {}

        for episode_data in all_annotations.values():
            segments = episode_data.get("label_info", {}).get("action_config", [])
            total_segments += len(segments)

            for segment in segments:
                skill = segment.get("skill", "Unknown")
                skills_count[skill] = skills_count.get(skill, 0) + 1

        return {
            "total_episodes_annotated": total_episodes,
            "total_segments": total_segments,
            "average_segments_per_episode": total_segments / total_episodes if total_episodes > 0 else 0,
            "skills_distribution": skills_count
        }

    def validate_data_integrity(self) -> Dict[str, Any]:
        """验证数据完整性"""
        issues = []
        warnings = []

        if not self.frames_path.exists():
            return {
                "valid": True,
                "issues": [],
                "warnings": ["frames.jsonl file does not exist"]
            }

        try:
            all_annotations = self.load_all()

            for episode_id, data in all_annotations.items():
                try:
                    # 尝试使用数据模型验证
                    EpisodeAnnotations(**data)
                except (ValueError, TypeError) as e:
                    issues.append(f"Episode {episode_id}: {e}")

        except Exception as e:
            issues.append(f"Failed to load annotations: {e}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

    def delete_episode(self, episode_id: int) -> Dict[str, Any]:
        """删除指定episode的标注数据"""
        with self._lock:
            if not self.frames_path.exists():
                return {
                    "ok": True,
                    "deleted": False,
                    "remaining": 0,
                    "message": "No annotations file exists"
                }

            # 读取现有数据
            all_annotations = self.load_all()

            if episode_id not in all_annotations:
                return {
                    "ok": True,
                    "deleted": False,
                    "remaining": len(all_annotations),
                    "message": f"Episode {episode_id} not found"
                }

            # 删除指定episode
            del all_annotations[episode_id]
            remaining_count = len(all_annotations)

            # 如果没有剩余数据，删除文件
            if remaining_count == 0:
                try:
                    self.frames_path.unlink()
                    return {
                        "ok": True,
                        "deleted": True,
                        "remaining": 0,
                        "file_removed": True,
                        "message": "Episode deleted and file removed (no remaining annotations)"
                    }
                except OSError as e:
                    self.logger.error(f"Failed to remove empty frames.jsonl: {e}")
                    return {
                        "ok": False,
                        "error": f"Failed to remove file: {e}"
                    }
            else:
                # 重写文件
                try:
                    data_array = list(all_annotations.values())
                    temp_dir = self.frames_path.parent
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                                   dir=temp_dir, delete=False,
                                                   suffix='.tmp') as temp_file:
                        temp_path = Path(temp_file.name)
                        try:
                            json.dump(data_array, temp_file, ensure_ascii=False, indent=2)
                            temp_file.flush()
                            temp_path.replace(self.frames_path)

                            return {
                                "ok": True,
                                "deleted": True,
                                "remaining": remaining_count,
                                "file_removed": False,
                                "message": f"Episode {episode_id} deleted, {remaining_count} annotations remaining"
                            }
                        except Exception:
                            if temp_path.exists():
                                temp_path.unlink()
                            raise
                except (IOError, OSError) as e:
                    self.logger.error(f"Failed to update frames.jsonl after deletion: {e}")
                    return {
                        "ok": False,
                        "error": f"Failed to update file: {e}"
                    }

    def delete_all(self) -> Dict[str, Any]:
        """删除所有标注数据"""
        with self._lock:
            if not self.frames_path.exists():
                return {
                    "ok": True,
                    "deleted_all": True,
                    "message": "No annotations file exists"
                }

            try:
                # 获取删除前的统计
                all_annotations = self.load_all()
                deleted_count = len(all_annotations)

                # 删除文件
                self.frames_path.unlink()

                return {
                    "ok": True,
                    "deleted_all": True,
                    "deleted_count": deleted_count,
                    "message": f"All {deleted_count} annotations deleted"
                }
            except OSError as e:
                self.logger.error(f"Failed to delete frames.jsonl: {e}")
                return {
                    "ok": False,
                    "error": f"Failed to delete file: {e}"
                }

    def check_skill_usage(self, skill_name: str) -> Dict[str, Any]:
        """检查技能是否被使用"""
        try:
            all_annotations = self.load_all()
            using_episodes = []

            for episode_id, annotation in all_annotations.items():
                action_config = annotation.get('label_info', {}).get('action_config', [])
                for segment in action_config:
                    if segment.get('skill') == skill_name:
                        using_episodes.append(episode_id)
                        break

            return {
                "skill_name": skill_name,
                "is_used": len(using_episodes) > 0,
                "using_episodes": using_episodes,
                "usage_count": len(using_episodes)
            }
        except Exception as e:
            self.logger.error(f"Error checking skill usage for {skill_name}: {e}")
            return {
                "skill_name": skill_name,
                "is_used": False,
                "using_episodes": [],
                "usage_count": 0,
                "error": str(e)
            }

    def rename_skill_globally(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """全局重命名技能，更新所有标注文件"""
        with self._lock:
            try:
                # 先检查新名称是否已存在
                if old_name == new_name:
                    return {
                        "ok": False,
                        "error": "新名称与原名称相同"
                    }

                if not new_name or not new_name.strip():
                    return {
                        "ok": False,
                        "error": "新名称不能为空"
                    }

                new_name = new_name.strip()

                # 加载所有标注数据
                all_annotations = self.load_all()
                updated_count = 0
                updated_episodes = []

                # 更新所有使用该技能的标注
                for episode_id, annotation in all_annotations.items():
                    action_config = annotation.get('label_info', {}).get('action_config', [])
                    episode_updated = False

                    for segment in action_config:
                        if segment.get('skill') == old_name:
                            segment['skill'] = new_name
                            episode_updated = True

                    if episode_updated:
                        updated_episodes.append(episode_id)
                        updated_count += 1

                if updated_count > 0:
                    # 重写文件
                    try:
                        data_array = list(all_annotations.values())
                        temp_dir = self.frames_path.parent
                        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                                       dir=temp_dir, delete=False,
                                                       suffix='.tmp') as temp_file:
                            temp_path = Path(temp_file.name)
                            try:
                                json.dump(data_array, temp_file, ensure_ascii=False, indent=2)
                                temp_file.flush()
                                temp_path.replace(self.frames_path)

                                return {
                                    "ok": True,
                                    "updated_count": updated_count,
                                    "updated_episodes": updated_episodes,
                                    "old_name": old_name,
                                    "new_name": new_name,
                                    "message": f"成功将技能 '{old_name}' 重命名为 '{new_name}'，更新了 {updated_count} 个episode"
                                }
                            except Exception:
                                if temp_path.exists():
                                    temp_path.unlink()
                                raise
                    except (IOError, OSError) as e:
                        self.logger.error(f"Failed to update annotations after skill rename: {e}")
                        return {
                            "ok": False,
                            "error": f"保存文件时出错: {e}"
                        }
                else:
                    return {
                        "ok": True,
                        "updated_count": 0,
                        "updated_episodes": [],
                        "old_name": old_name,
                        "new_name": new_name,
                        "message": f"技能 '{old_name}' 没有被使用，无需更新"
                    }

            except Exception as e:
                self.logger.error(f"Error renaming skill from {old_name} to {new_name}: {e}")
                return {
                    "ok": False,
                    "error": f"重命名技能时出错: {e}"
                }

    def delete_skill_if_unused(self, skill_name: str) -> Dict[str, Any]:
        """删除技能（仅限未使用的）"""
        try:
            # 检查技能是否被使用
            usage_check = self.check_skill_usage(skill_name)

            if "error" in usage_check:
                return {
                    "ok": False,
                    "error": usage_check["error"]
                }

            if usage_check["is_used"]:
                return {
                    "ok": False,
                    "error": f"技能 '{skill_name}' 正在被使用，无法删除",
                    "using_episodes": usage_check["using_episodes"],
                    "usage_count": usage_check["usage_count"]
                }

            # 这里实际上不需要删除什么，因为技能列表是动态生成的
            # 只要没有标注使用这个技能，它就不会出现在历史列表中
            return {
                "ok": True,
                "skill_name": skill_name,
                "message": f"技能 '{skill_name}' 未被使用，已从列表中移除"
            }

        except Exception as e:
            self.logger.error(f"Error deleting skill {skill_name}: {e}")
            return {
                "ok": False,
                "error": f"删除技能时出错: {e}"
            }