#!/usr/bin/env python3
"""
LeRobot数据集Parquet文件分析工具

用于读取和分析LeRobot数据集中的parquet文件，
提供详细的数据结构分析和统计信息。

使用方法:
    python analyze_parquet.py [parquet_file_path]
    
如果不提供路径，将使用默认的第一个episode文件。
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


def analyze_parquet_file(file_path):
    """
    分析parquet文件的数据结构
    
    Args:
        file_path (str): parquet文件路径
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        print('=' * 50)
        print('Parquet文件数据结构分析')
        print('=' * 50)
        print(f'文件路径: {file_path}')
        print()

        # 基本信息
        print('=== 基本信息 ===')
        print(f'数据形状: {df.shape}')
        print(f'行数: {df.shape[0]}')
        print(f'列数: {df.shape[1]}')
        print()

        # 列名和数据类型
        print('=== 列名和数据类型 ===')
        for i, (col, dtype) in enumerate(df.dtypes.items()):
            print(f'{i+1:2d}. {col:<30} {dtype}')
        print()

        # 前5行数据
        print('=== 前5行数据 ===')
        print(df.head(-5))
        print()

        # 数据统计信息
        print('=== 数值列统计信息 ===')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print('没有数值类型的列')
        print()

        # 缺失值统计
        print('=== 缺失值统计 ===')
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print('没有缺失值')
        print()

        # 内存使用情况
        print('=== 内存使用情况 ===')
        print(df.info(memory_usage='deep'))
        print()

        # 详细分析action和observation.state列（如果存在）
        if 'action' in df.columns:
            analyze_array_column(df, 'action')
        
        if 'observation.state' in df.columns:
            analyze_array_column(df, 'observation.state')
        
        # 时间序列信息（如果存在timestamp列）
        if 'timestamp' in df.columns:
            analyze_timestamp_info(df)
        
        # episode信息
        if 'episode_index' in df.columns:
            analyze_episode_info(df)
            
    except Exception as e:
        print(f"读取文件时出错: {e}")


def analyze_array_column(df, column_name):
    """
    分析包含数组数据的列
    
    Args:
        df (pd.DataFrame): 数据框
        column_name (str): 列名
    """
    print(f'=== {column_name}列详细分析 ===')
    
    if column_name not in df.columns:
        print(f'{column_name}列不存在')
        return
    
    first_data = df[column_name].iloc[0]
    print(f'{column_name}列数据类型: {type(first_data)}')
    print(f'第一个{column_name}数据: {first_data}')
    print(f'{column_name}数据长度: {len(first_data)}')
    print(f'{column_name}数据形状: {np.array(first_data).shape}')
    print()
    
    # 统计信息
    try:
        array_data = np.array(df[column_name].tolist())
        print(f'{column_name}数组形状: {array_data.shape}')
        print(f'{column_name}各维度统计:')
        
        max_dims_to_show = min(10, array_data.shape[1])
        for i in range(max_dims_to_show):
            min_val = array_data[:, i].min()
            max_val = array_data[:, i].max()
            mean_val = array_data[:, i].mean()
            std_val = array_data[:, i].std()
            print(f'  维度{i}: min={min_val:.3f}, max={max_val:.3f}, '
                  f'mean={mean_val:.3f}, std={std_val:.3f}')
        
        if array_data.shape[1] > max_dims_to_show:
            print(f'  ... (还有{array_data.shape[1] - max_dims_to_show}个维度)')
        print()
        
    except Exception as e:
        print(f'分析{column_name}数组数据时出错: {e}')
        print()


def analyze_timestamp_info(df):
    """
    分析时间戳信息
    
    Args:
        df (pd.DataFrame): 数据框
    """
    print('=== 时间序列信息 ===')
    
    if 'timestamp' not in df.columns:
        print('没有timestamp列')
        return
    
    timestamps = df['timestamp']
    time_diff = timestamps.diff().dropna()
    
    print(f'时间戳范围: {timestamps.min():.3f} - {timestamps.max():.3f} 秒')
    print(f'总时长: {timestamps.max() - timestamps.min():.3f} 秒')
    print(f'平均时间间隔: {time_diff.mean():.6f} 秒')
    print(f'时间间隔标准差: {time_diff.std():.6f} 秒')
    print(f'估计采样频率: {1/time_diff.mean():.1f} Hz')
    print()


def analyze_episode_info(df):
    """
    分析episode相关信息
    
    Args:
        df (pd.DataFrame): 数据框
    """
    print('=== Episode信息 ===')
    
    if 'episode_index' in df.columns:
        print(f'episode_index: {df["episode_index"].unique()}')
    
    if 'task_index' in df.columns:
        print(f'task_index: {df["task_index"].unique()}')
    
    if 'frame_index' in df.columns:
        print(f'frame_index范围: {df["frame_index"].min()} - {df["frame_index"].max()}')
    
    if 'index' in df.columns:
        print(f'index范围: {df["index"].min()} - {df["index"].max()}')
    
    print()


def main():
    """主函数"""
    # 默认文件路径
    default_path = "/home/chenqingyu/robot/lerobot_data/datasets/grasp_dataset/data/chunk-000/episode_000001.parquet"
    
    # 从命令行参数获取文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path
        print(f"使用默认文件路径: {file_path}")
        print()
    
    # 分析文件
    analyze_parquet_file(file_path)


if __name__ == "__main__":
    main()
