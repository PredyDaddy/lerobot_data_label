"""
数据加载器模块
功能: 统一处理数据集加载、Episode数据提取、CSV格式转换
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Iterator
import json
import pandas as pd
import numpy as np
from io import StringIO
import csv


class IterableNamespace(SimpleNamespace):
    """
    支持迭代的命名空间对象
    提供字典和点号双重访问方式
    """
    def __init__(self, dictionary: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if dictionary is not None:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, IterableNamespace(value))
                else:
                    setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def items(self):
        return vars(self).items()

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()


class LocalDatasetLoader:
    """
    本地数据集加载器
    专门处理grasp_dataset格式的数据
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.info = self._load_info()
        self.episodes_list = self._load_episodes()
        self.episodes_metadata = self._load_episodes_metadata()

    def _load_info(self) -> IterableNamespace:
        """加载数据集基本信息"""
        info_path = self.dataset_path / "meta" / "info.json"
        with open(info_path, 'r') as f:
            info_data = json.load(f)
        return IterableNamespace(info_data)

    def _load_episodes(self) -> List[int]:
        """加载episode列表"""
        return list(range(self.info.total_episodes))

    def _load_episodes_metadata(self) -> Dict[int, Dict[str, Any]]:
        """加载episodes.jsonl文件中的元数据"""
        episodes_metadata = {}
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"

        if not episodes_path.exists():
            # 如果episodes.jsonl不存在，返回空字典
            return episodes_metadata

        try:
            with open(episodes_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            episode_data = json.loads(line)
                            episode_index = episode_data.get("episode_index")
                            if episode_index is not None:
                                episodes_metadata[episode_index] = episode_data
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num} in episodes.jsonl: {e}")
        except (IOError, OSError) as e:
            print(f"Warning: Failed to read episodes.jsonl: {e}")

        return episodes_metadata

    def get_episode_tasks(self, episode_index: int) -> List[str]:
        """获取指定episode的tasks列表"""
        episode_data = self.episodes_metadata.get(episode_index, {})
        return episode_data.get("tasks", [])

    def load_episode_data(self, episode_index: int) -> str:
        """
        加载单个episode的时间序列数据，返回CSV格式字符串
        供前端Dygraphs使用
        """
        # 构建数据文件路径
        episode_chunk = episode_index // self.info.chunks_size
        data_path = self.dataset_path / self.info.data_path.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index
        )

        # 读取parquet文件
        df = pd.read_parquet(data_path)

        # 选择数值型列用于可视化
        numeric_columns = []
        header = ["timestamp"]

        # 选择float32和int32类型的列，但排除timestamp
        selected_columns = [col for col, ft in self.info.features.items()
                          if ft["dtype"] in ["float32", "int32"] and col != "timestamp"]

        for column_name in selected_columns:
            shape = self.info.features[column_name]["shape"]
            shape_dim = len(shape)

            # 只处理一维数据，跳过多维数据
            if shape_dim == 1:
                if "names" in self.info.features[column_name] and self.info.features[column_name]["names"]:
                    column_names = self.info.features[column_name]["names"]
                    header.extend(column_names)
                    numeric_columns.append(column_name)
                else:
                    # 如果没有names，使用默认命名
                    dim_state = shape[0]
                    column_names = [f"{column_name}_{i}" for i in range(dim_state)]
                    header.extend(column_names)
                    numeric_columns.append(column_name)

        # 构建CSV数据
        selected_columns = ["timestamp"] + numeric_columns
        data_subset = df[selected_columns]

        # 转换为CSV字符串
        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(header)

        # 写入数据行
        for _, row in data_subset.iterrows():
            csv_row = [row["timestamp"]]
            for col in numeric_columns:
                if isinstance(row[col], np.ndarray):
                    csv_row.extend(row[col].tolist())
                else:
                    csv_row.append(row[col])
            csv_writer.writerow(csv_row)

        return csv_buffer.getvalue()

    def get_episode_info(self, episode_index: int) -> Dict[str, Any]:
        """获取episode基本信息"""
        return {
            "episode_id": episode_index,
            "total_episodes": self.info.total_episodes,
            "fps": self.info.fps
        }

    def get_columns_info(self, episode_index: int) -> List[Dict[str, Any]]:
        """获取列信息用于前端显示"""
        columns = []

        # 选择float32和int32类型的列，但排除timestamp
        selected_columns = [col for col, ft in self.info.features.items()
                          if ft["dtype"] in ["float32", "int32"] and col != "timestamp"]

        for column_name in selected_columns:
            shape = self.info.features[column_name]["shape"]
            shape_dim = len(shape)

            # 只处理一维数据
            if shape_dim == 1:
                if "names" in self.info.features[column_name] and self.info.features[column_name]["names"]:
                    column_names = self.info.features[column_name]["names"]
                else:
                    dim_state = shape[0]
                    column_names = [f"{column_name}_{i}" for i in range(dim_state)]

                columns.append({"key": column_name, "value": column_names})

        return columns

    def get_ignored_columns(self) -> List[str]:
        """获取被忽略的列（多维数据）"""
        ignored_columns = []

        for column_name, feature in self.info.features.items():
            if feature["dtype"] in ["float32", "int32"] and column_name != "timestamp":
                shape = feature["shape"]
                shape_dim = len(shape)
                if shape_dim > 1:
                    ignored_columns.append(column_name)

        return ignored_columns

    def get_video_keys(self) -> List[str]:
        """
        动态获取数据集中所有视频特征键
        参照原项目: video_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "video"]
        """
        video_keys = []
        for key, feature in self.info.features.items():
            # 处理IterableNamespace和普通字典
            if isinstance(feature, IterableNamespace):
                dtype = getattr(feature, 'dtype', None)
            else:
                dtype = feature.get("dtype")

            if dtype == "video":
                video_keys.append(key)
        return video_keys