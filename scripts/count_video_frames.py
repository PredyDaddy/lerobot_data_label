#!/usr/bin/env python3
"""
使用OpenCV计算视频帧数的脚本
支持多种方法：CV_CAP_PROP_FRAME_COUNT属性和逐帧读取计数
"""

import cv2
import os
import time
import argparse


def count_frames_by_property(video_path):
    """
    方法1: 使用CV_CAP_PROP_FRAME_COUNT属性获取帧数
    这是最快的方法，但对某些视频格式可能不准确
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "无法打开视频文件"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return {
        'method': 'CV_CAP_PROP_FRAME_COUNT',
        'frame_count': frame_count,
        'fps': fps,
        'duration': duration
    }, None


def count_frames_by_iteration(video_path):
    """
    方法2: 逐帧读取计数
    这是最准确的方法，但速度较慢
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "无法打开视频文件"
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("开始逐帧计数...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # 每1000帧显示一次进度
        if frame_count % 1000 == 0:
            print(f"已读取 {frame_count} 帧...")
    
    end_time = time.time()
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return {
        'method': 'Frame Iteration',
        'frame_count': frame_count,
        'fps': fps,
        'duration': duration,
        'processing_time': end_time - start_time
    }, None


def get_video_info(video_path):
    """获取视频的基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "无法打开视频文件"
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'codec': ''.join([chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF) for i in range(4)])
    }
    
    cap.release()
    return info, None


def main():
    parser = argparse.ArgumentParser(description='使用OpenCV计算视频帧数')
    parser.add_argument('video_path', nargs='?', 
                       default='/home/chenqingyu/robot/lerobot_data/datasets/grasp_dataset/videos/chunk-000/observation.images.laptop/episode_000001.mp4',
                       help='视频文件路径')
                       
    parser.add_argument('--method', choices=['property', 'iteration', 'both'], default='both',
                       help='计数方法: property(属性), iteration(逐帧), both(两种方法)')
    parser.add_argument('--info', action='store_true', help='显示视频详细信息')
    
    args = parser.parse_args()
    video_path = args.video_path
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    print(f"分析视频: {video_path}")
    print(f"文件大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    print("-" * 60)
    
    # 获取视频基本信息
    if args.info:
        info, error = get_video_info(video_path)
        if error:
            print(f"错误: {error}")
            return
        
        print("视频信息:")
        print(f"  分辨率: {info['width']} x {info['height']}")
        print(f"  FPS: {info['fps']:.2f}")
        print(f"  编码: {info['codec']}")
        print("-" * 60)
    
    # 方法1: 使用属性获取帧数
    if args.method in ['property', 'both']:
        print("方法1: 使用CV_CAP_PROP_FRAME_COUNT属性")
        result1, error1 = count_frames_by_property(video_path)
        if error1:
            print(f"错误: {error1}")
        else:
            print(f"  帧数: {result1['frame_count']}")
            print(f"  FPS: {result1['fps']:.2f}")
            print(f"  时长: {result1['duration']:.2f} 秒")
        print()
    
    # 方法2: 逐帧计数
    if args.method in ['iteration', 'both']:
        print("方法2: 逐帧读取计数")
        result2, error2 = count_frames_by_iteration(video_path)
        if error2:
            print(f"错误: {error2}")
        else:
            print(f"  帧数: {result2['frame_count']}")
            print(f"  FPS: {result2['fps']:.2f}")
            print(f"  时长: {result2['duration']:.2f} 秒")
            print(f"  处理时间: {result2['processing_time']:.2f} 秒")
        print()
    
    # 比较两种方法的结果
    if args.method == 'both':
        if not error1 and not error2:
            diff = abs(result1['frame_count'] - result2['frame_count'])
            print("结果比较:")
            print(f"  属性方法帧数: {result1['frame_count']}")
            print(f"  逐帧计数帧数: {result2['frame_count']}")
            print(f"  差异: {diff} 帧")
            if diff == 0:
                print("  ✅ 两种方法结果一致")
            else:
                print(f"  ⚠️  两种方法结果不一致，建议使用逐帧计数结果")


if __name__ == "__main__":
    main()
