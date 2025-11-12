"""
Flask Web应用
功能: Web路由、API端点、静态文件服务
"""

import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request, redirect, session

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.data_loader import LocalDatasetLoader, IterableNamespace
from core.config import DatasetConfig, AppConfig
from core.video_handler import VideoHandler
from core.annotation_store import AnnotationStore
from core.dataset_manager import DatasetManager


def convert_namespace_to_dict(obj):
    """递归转换IterableNamespace对象为普通字典"""
    if isinstance(obj, IterableNamespace):
        return {key: convert_namespace_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, dict):
        return {key: convert_namespace_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_namespace_to_dict(item) for item in obj]
    else:
        return obj


def create_app(dataset_path: str = None, config_path: str = "config.yaml") -> Flask:
    """
    Flask应用工厂函数
    """
    app = Flask(__name__,
                static_folder=None,  # 我们将手动处理静态文件
                template_folder=str(Path(__file__).parent.parent / "templates"))

    # 配置Flask应用
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # 禁用缓存用于开发
    app.secret_key = "lerobot_annotation_platform_secret_key"  # 会话管理

    # 初始化配置和核心组件
    app_config = AppConfig(config_path)
    if dataset_path is None:
        dataset_path = app_config.get_dataset_path()

    dataset_config = DatasetConfig(dataset_path)
    data_loader = LocalDatasetLoader(dataset_path)
    video_handler = VideoHandler(dataset_config, data_loader, app_config)
    annotation_store = AnnotationStore(dataset_config)

    # 初始化数据集管理器（包含静止帧删除服务）
    dataset_manager = DatasetManager(dataset_path, app_config)

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    @app.before_request
    def _ensure_valid_dataset_path():
        """确保会话中的数据集路径有效；无效则回退到默认路径。
        避免用户移动/重命名数据集后仍使用旧的 session 路径导致 404。
        """
        try:
            p = session.get('dataset_path')
            if p:
                pp = Path(p)
                valid = pp.exists() and (pp / "meta" / "info.json").exists() and (pp / "videos").exists()
                if not valid:
                    # 清除无效路径，路由中的 session.get('dataset_path', dataset_path) 将自动回退
                    session.pop('dataset_path', None)
        except Exception as e:
            logger.warning(f"Failed to validate session dataset_path: {e}")

    @app.route("/")
    def homepage():
        """主页 - 数据集选择页面"""
        # 检查会话中是否有数据集路径
        session_dataset_path = session.get('dataset_path')
        if session_dataset_path and Path(session_dataset_path).exists():
            current_dataset = session_dataset_path
        else:
            current_dataset = dataset_path

        return render_template(
            "home.html",
            current_dataset_path=current_dataset,
            dataset_name=Path(current_dataset).name
        )

    @app.route("/episode_<int:episode_id>")
    def show_episode(episode_id: int):
        """显示指定episode的详情页"""
        try:
            # 获取当前使用的数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)
            current_video_handler = VideoHandler(current_dataset_config, current_data_loader)

            # 验证episode_id有效性
            if episode_id < 0 or episode_id >= current_data_loader.info.total_episodes:
                return f"Episode {episode_id} not found. Valid range: 0-{current_data_loader.info.total_episodes-1}", 404

            # 获取episode数据
            episode_data_csv_str = current_data_loader.load_episode_data(episode_id)
            columns = current_data_loader.get_columns_info(episode_id)
            ignored_columns = current_data_loader.get_ignored_columns()

            # 获取数据集信息
            dataset_info = {
                "repo_id": f"local/{Path(current_dataset_path).name}",
                "num_samples": current_data_loader.info.total_frames,
                "num_episodes": current_data_loader.info.total_episodes,
                "fps": current_data_loader.info.fps
            }

            # 获取视频信息
            videos_info = current_video_handler.get_video_info(episode_id)

            # 添加语言指令 (如果有的话)
            if videos_info:
                videos_info[0]["language_instruction"] = None  # 简化版本不包含任务信息

            # episode列表
            episodes = list(range(current_data_loader.info.total_episodes))

            return render_template(
                "visualize_dataset_template.html",
                episode_id=episode_id,
                episodes=episodes,
                dataset_info=dataset_info,
                videos_info=videos_info,
                episode_data_csv_str=episode_data_csv_str,
                columns=columns,
                ignored_columns=ignored_columns
            )

        except FileNotFoundError as e:
            logger.error(f"Episode data file not found: {e}")
            return f"Episode {episode_id} data not found", 404
        except Exception as e:
            logger.error(f"Error loading episode {episode_id}: {e}")
            return f"Internal server error: {str(e)}", 500

    @app.route("/api/dataset/info")
    def api_dataset_info():
        """API: 获取数据集基本信息"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)

            # 使用递归转换函数处理IterableNamespace
            info = {
                "total_episodes": current_data_loader.info.total_episodes,
                "total_frames": current_data_loader.info.total_frames,
                "fps": current_data_loader.info.fps,
                "robot_type": current_data_loader.info.robot_type,
                "codebase_version": current_data_loader.info.codebase_version,
                "features": convert_namespace_to_dict(current_data_loader.info.features)
            }
            return jsonify(info)
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episodes")
    def api_episodes_list():
        """API: 获取episodes列表"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)

            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', app_config.get_episodes_per_page()))

            start = (page - 1) * per_page
            end = start + per_page

            all_episodes = current_data_loader.episodes_list
            episodes_slice = all_episodes[start:end]

            return jsonify({
                "episodes": [{"id": ep_id} for ep_id in episodes_slice],
                "total": len(all_episodes),
                "page": page,
                "per_page": per_page,
                "total_pages": (len(all_episodes) + per_page - 1) // per_page
            })
        except Exception as e:
            logger.error(f"Error getting episodes list: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/data")
    def api_episode_data(episode_id: int):
        """API: 获取episode时间序列数据 (CSV格式)"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)

            if episode_id < 0 or episode_id >= current_data_loader.info.total_episodes:
                return jsonify({"error": f"Episode {episode_id} not found"}), 404

            csv_data = current_data_loader.load_episode_data(episode_id)
            return csv_data, 200, {'Content-Type': 'text/csv'}
        except FileNotFoundError:
            return jsonify({"error": f"Episode {episode_id} data file not found"}), 404
        except Exception as e:
            logger.error(f"Error getting episode {episode_id} data: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/videos")
    def api_episode_videos(episode_id: int):
        """API: 获取episode视频信息"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)
            current_video_handler = VideoHandler(current_dataset_config, current_data_loader)

            if episode_id < 0 or episode_id >= current_data_loader.info.total_episodes:
                return jsonify({"error": f"Episode {episode_id} not found"}), 404

            videos_info = current_video_handler.get_video_info(episode_id)

            response = jsonify({"videos": videos_info})

            # 添加强制禁缓存头部
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            return response
        except Exception as e:
            logger.error(f"Error getting episode {episode_id} videos: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/motion_detection")
    def api_episode_motion_detection(episode_id: int):
        """API: 获取episode的静止帧检测状态"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)
            current_video_handler = VideoHandler(current_dataset_config, current_data_loader, app_config)

            status = current_video_handler.get_motion_detection_status()
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error getting motion detection status for episode {episode_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/static_frames/<video_key>")
    def api_episode_static_frames(episode_id: int, video_key: str):
        """API: 获取episode指定视频的静止帧检测结果"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_manager = DatasetManager(current_dataset_path, app_config)

            # 获取查询参数
            start_frame = request.args.get('start_frame', 0, type=int)
            end_frame = request.args.get('end_frame', None, type=int)

            # 获取动态检测参数
            threshold = request.args.get('threshold', None, type=float)
            method = request.args.get('method', None, type=str)
            min_static_frames = request.args.get('min_static_frames', None, type=int)

            # 检测静止帧
            results = current_dataset_manager.static_frames.detect_static_frames(
                episode_id, video_key, start_frame, end_frame,
                threshold=threshold, method=method, min_static_frames=min_static_frames
            )

            # 转换结果为JSON格式
            json_results = []
            for result in results:
                json_results.append({
                    "frame_index": int(result.frame_index),
                    "is_static": bool(result.is_static),
                    "motion_score": float(result.motion_score),
                    "timestamp": float(result.timestamp) if result.timestamp is not None else None
                })

            return jsonify({
                "episode_id": episode_id,
                "video_key": video_key,
                "results": json_results,
                "total_frames": len(json_results),
                "static_frames": sum(1 for r in results if r.is_static)
            })
        except Exception as e:
            logger.error(f"Error detecting static frames for episode {episode_id}, video {video_key}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/static_segments/<video_key>")
    def api_episode_static_segments(episode_id: int, video_key: str):
        """API: 获取episode指定视频的静止片段"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_manager = DatasetManager(current_dataset_path, app_config)

            # 获取动态检测参数
            threshold = request.args.get('threshold', None, type=float)
            method = request.args.get('method', None, type=str)
            min_static_frames = request.args.get('min_static_frames', None, type=int)

            # 获取静止片段
            segments = current_dataset_manager.static_frames.get_static_segments(
                episode_id, video_key,
                threshold=threshold, method=method, min_static_frames=min_static_frames
            )

            # 转换为JSON格式
            json_segments = []
            for start_frame, end_frame in segments:
                json_segments.append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": end_frame - start_frame + 1
                })

            return jsonify({
                "episode_id": episode_id,
                "video_key": video_key,
                "segments": json_segments,
                "total_segments": len(json_segments)
            })
        except Exception as e:
            logger.error(f"Error getting static segments for episode {episode_id}, video {video_key}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/motion_stats/<video_key>")
    def api_episode_motion_stats(episode_id: int, video_key: str):
        """API: 获取episode指定视频的运动统计信息"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_manager = DatasetManager(current_dataset_path, app_config)

            # 获取动态检测参数
            threshold = request.args.get('threshold', None, type=float)
            method = request.args.get('method', None, type=str)
            min_static_frames = request.args.get('min_static_frames', None, type=int)

            # 获取运动统计
            stats = current_dataset_manager.static_frames.get_motion_statistics(
                episode_id, video_key,
                threshold=threshold, method=method, min_static_frames=min_static_frames
            )

            return jsonify({
                "episode_id": episode_id,
                "video_key": video_key,
                "statistics": stats
            })
        except Exception as e:
            logger.error(f"Error getting motion statistics for episode {episode_id}, video {video_key}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/motion_detection/clear_cache", methods=["POST"])
    def api_clear_motion_cache():
        """API: 清除静止帧检测缓存"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_manager = DatasetManager(current_dataset_path, app_config)

            current_dataset_manager.static_frames.clear_motion_cache()

            return jsonify({"ok": True, "message": "Motion detection cache cleared"})
        except Exception as e:
            logger.error(f"Error clearing motion detection cache: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/delete_frames", methods=["POST"])
    def api_delete_frames(episode_id: int):
        """统一的帧删除API"""
        try:
            data = request.get_json()
            deletion_type = data.get('deletion_type')
            video_key = data.get('video_key')
            create_backup = data.get('create_backup', True)

            if not deletion_type:
                return jsonify({"success": False, "error": "deletion_type is required"}), 400

            if not video_key:
                return jsonify({"success": False, "error": "video_key is required"}), 400

            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_manager = DatasetManager(current_dataset_path, app_config)

            # 处理视频键，遵循统一逻辑
            if video_key == "all":
                video_keys = "all"
            else:
                video_keys = [video_key]

            # 根据删除类型确定要删除的帧索引
            frame_indices = []

            if deletion_type == "static_frames":
                # 静止帧删除：使用plan_pruning检测静止帧
                plan = current_dataset_manager.static_frames.plan_pruning(
                    episode_id, video_keys=video_keys
                )

                if "error" in plan:
                    return jsonify({"success": False, "error": plan["error"]}), 500

                if not plan["frame_indices"]:
                    return jsonify({
                        "success": True,
                        "episode_id": episode_id,
                        "requested": 0,
                        "deleted": 0,
                        "failed_indices": [],
                        "per_view": {},
                        "message": "没有检测到静止帧"
                    })

                frame_indices = plan["frame_indices"]

            elif deletion_type == "single_frame":
                # 单帧删除
                frame_index = data.get('frame_index')
                if frame_index is None:
                    return jsonify({"success": False, "error": "frame_index is required for single_frame deletion"}), 400
                frame_indices = [frame_index]

            elif deletion_type == "frame_range":
                # 帧段删除
                start_frame = data.get('start_frame')
                end_frame = data.get('end_frame')

                if start_frame is None or end_frame is None:
                    return jsonify({"success": False, "error": "start_frame and end_frame are required for frame_range deletion"}), 400

                if start_frame >= end_frame:
                    return jsonify({"success": False, "error": "start_frame must be less than end_frame"}), 400

                frame_indices = list(range(start_frame, end_frame + 1))

            else:
                return jsonify({"success": False, "error": f"Unknown deletion_type: {deletion_type}"}), 400

            # 统一调用delete_frames执行删除
            result = current_dataset_manager.static_frames.delete_frames(
                episode_id, frame_indices, video_keys=video_keys,
                create_backup=create_backup
            )

            return jsonify(result)

        except Exception as e:
            logger.error(f"Error in unified delete frames API for episode {episode_id}: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/frame_count", methods=["GET"])
    def api_get_frame_count(episode_id: int):
        """获取episode的准确帧数（从数据源而非视频duration）"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)

            # 直接构造Parquet文件路径，避免依赖DatasetManager的内部方法
            chunk_id = episode_id // 1000
            chunk_dir = f"chunk-{chunk_id:03d}"
            episode_filename = f"episode_{episode_id:06d}.parquet"
            parquet_path = Path(current_dataset_path) / "data" / chunk_dir / episode_filename

            if not parquet_path.exists():
                return jsonify({"success": False, "error": "Parquet文件不存在"}), 404

            # 读取Parquet文件获取实际帧数
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            total_frames = len(df)

            response = jsonify({
                "success": True,
                "episode_id": episode_id,
                "total_frames": total_frames
            })

            # 添加强制禁缓存头部
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            return response

        except Exception as e:
            logger.error(f"Error getting frame count for episode {episode_id}: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/refresh_data")
    def api_refresh_episode_data(episode_id: int):
        """删除后刷新episode数据API - 强制从文件重读"""
        try:
            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)

            # 重新创建DataLoader实例，避免使用缓存
            current_data_loader = LocalDatasetLoader(current_dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_video_handler = VideoHandler(current_dataset_config, current_data_loader)

            # 验证episode_id有效性
            if episode_id < 0 or episode_id >= current_data_loader.info.total_episodes:
                return jsonify({"success": False, "error": f"Episode {episode_id} not found"}), 404

            # 强制重新读取parquet文件数据
            episode_data_csv = current_data_loader.load_episode_data(episode_id)

            # 重新获取视频信息（总帧数等）
            videos_info = current_video_handler.get_video_info(episode_id)

            # 重新读取数据集统计信息
            dataset_info = {
                "repo_id": f"local/{Path(current_dataset_path).name}",
                "num_samples": current_data_loader.info.total_frames,
                "num_episodes": current_data_loader.info.total_episodes,
                "fps": current_data_loader.info.fps
            }

            response = jsonify({
                "success": True,
                "episode_data_csv": episode_data_csv,
                "videos_info": videos_info,
                "dataset_info": dataset_info,
                "columns": current_data_loader.get_columns_info(episode_id)
            })

            # 添加强制禁缓存头部
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            return response

        except Exception as e:
            logger.error(f"Error refreshing episode {episode_id} data: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>/video_frame_info/<video_key>")
    def api_get_video_frame_info(episode_id: int, video_key: str):
        """获取基于OpenCV的精确视频帧信息"""
        try:
            import cv2

            # 获取当前数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_manager = DatasetManager(current_dataset_path, app_config)

            # 获取视频文件路径
            video_path = current_dataset_manager.static_frames._get_video_file_path(episode_id, video_key)
            if not video_path or not video_path.exists():
                return jsonify({"success": False, "error": f"视频文件不存在: {video_key}"}), 404

            # 使用OpenCV获取精确视频信息
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return jsonify({"success": False, "error": f"无法打开视频文件: {video_path}"}), 500

            try:
                # 获取总帧数和帧率
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0

                # 可选：获取当前时间对应的帧号
                current_time = request.args.get('current_time', 0, type=float)
                current_frame = int(current_time * fps) if fps > 0 else 0
                current_frame = max(0, min(current_frame, total_frames - 1))

                return jsonify({
                    "success": True,
                    "episode_id": episode_id,
                    "video_key": video_key,
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration": duration,
                    "current_frame": current_frame,
                    "current_time": current_time
                })

            finally:
                cap.release()

        except ImportError:
            logger.error("OpenCV (cv2) not available")
            return jsonify({"success": False, "error": "OpenCV未安装，无法提供精确帧信息"}), 500
        except Exception as e:
            logger.error(f"Error getting video frame info for episode {episode_id}, video {video_key}: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/videos/<path:filename>")
    def serve_video(filename):
        """服务视频文件"""
        try:
            # 获取当前数据集路径（从会话中或使用默认值）
            current_dataset_path = session.get('dataset_path', dataset_path)
            video_dir = Path(current_dataset_path) / "videos"
            file_path = video_dir / filename

            # 验证文件存在
            if not file_path.exists():
                logger.error(f"Video file not found: {file_path}")
                return "Video not found", 404

            return send_from_directory(video_dir, filename)
        except Exception as e:
            logger.error(f"Error serving video {filename}: {e}")
            return "Video not found", 404

    @app.route("/static/<path:filename>")
    def serve_static(filename):
        """服务静态文件"""
        try:
            static_dir = Path(__file__).parent.parent / "static"
            if static_dir.exists():
                return send_from_directory(static_dir, filename)
            else:
                return "Static file not found", 404
        except Exception as e:
            logger.error(f"Error serving static file {filename}: {e}")
            return "Static file not found", 404

    def redirect_to_episode(episode_id: int):
        """重定向到指定episode"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LeRobot Visualizer</title>
            <meta http-equiv="refresh" content="0; url=/episode_{episode_id}">
        </head>
        <body>
            <p>Redirecting to episode {episode_id}...</p>
            <p><a href="/episode_{episode_id}">Click here if you are not redirected</a></p>
        </body>
        </html>
        """

    # ========== 标注功能相关路由 ==========

    @app.route("/annotate/<int:episode_id>")
    def annotate_episode(episode_id: int):
        """标注页面"""
        try:
            # 获取当前使用的数据集配置
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)
            current_video_handler = VideoHandler(current_dataset_config, current_data_loader)

            # 验证episode_id有效性
            if episode_id < 0 or episode_id >= current_data_loader.info.total_episodes:
                return f"Episode {episode_id} not found. Valid range: 0-{current_data_loader.info.total_episodes-1}", 404

            # 获取数据集信息
            dataset_info = {
                "repo_id": f"local/{Path(current_dataset_path).name}",
                "num_samples": current_data_loader.info.total_frames,
                "num_episodes": current_data_loader.info.total_episodes,
                "fps": current_data_loader.info.fps
            }

            # 获取视频信息
            videos_info = current_video_handler.get_video_info(episode_id)

            # 获取episode的tasks数据
            episode_tasks = current_data_loader.get_episode_tasks(episode_id)

            # episode列表
            episodes = list(range(current_data_loader.info.total_episodes))

            return render_template(
                "annotate.html",
                episode_id=episode_id,
                episodes=episodes,
                dataset_info=dataset_info,
                videos_info=videos_info,
                dataset_path=current_dataset_path,
                episode_tasks=episode_tasks
            )

        except FileNotFoundError as e:
            logger.error(f"Episode annotation file not found: {e}")
            return f"Episode {episode_id} not found", 404
        except Exception as e:
            logger.error(f"Error loading annotation page for episode {episode_id}: {e}")
            return f"Internal server error: {str(e)}", 500

    @app.route("/api/dataset/select", methods=["POST"])
    def api_dataset_select():
        """API: 选择数据集"""
        try:
            data = request.get_json()
            if not data or 'dataset_path' not in data:
                return jsonify({"ok": False, "error": "dataset_path is required"}), 400

            new_dataset_path = data['dataset_path']
            dataset_path_obj = Path(new_dataset_path)

            # 验证数据集路径
            if not dataset_path_obj.exists():
                return jsonify({"ok": False, "error": "Dataset path does not exist"}), 400

            info_path = dataset_path_obj / "meta" / "info.json"
            videos_path = dataset_path_obj / "videos"

            if not info_path.exists():
                return jsonify({"ok": False, "error": "meta/info.json not found"}), 400

            if not videos_path.exists():
                return jsonify({"ok": False, "error": "videos/ directory not found"}), 400

            # 尝试加载数据集信息
            temp_loader = LocalDatasetLoader(new_dataset_path)

            # 保存到会话
            session['dataset_path'] = new_dataset_path

            return jsonify({
                "ok": True,
                "dataset": {
                    "path": new_dataset_path,
                    "name": dataset_path_obj.name,
                    "total_episodes": temp_loader.info.total_episodes,
                    "fps": temp_loader.info.fps
                }
            })

        except Exception as e:
            logger.error(f"Error selecting dataset: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/dataset/current")
    def api_dataset_current():
        """API: 获取当前数据集"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            return jsonify({
                "path": current_dataset_path,
                "name": Path(current_dataset_path).name,
                "ok": True
            })
        except Exception as e:
            logger.error(f"Error getting current dataset: {e}")
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/annotations/stats")
    def api_get_annotations_stats():
        """API: 获取所有episodes的标注统计"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)
            current_data_loader = LocalDatasetLoader(current_dataset_path)

            # 获取所有标注数据
            all_annotations = current_annotation_store.load_all()

            # 计算每个episode的段落数量
            stats = {}
            annotated_episodes_count = 0
            total_segments_count = 0

            for episode_id, annotation_data in all_annotations.items():
                action_config = annotation_data.get('label_info', {}).get('action_config', [])
                segment_count = len(action_config)
                stats[episode_id] = segment_count

                # 计算全局统计
                if segment_count > 0:
                    annotated_episodes_count += 1
                    total_segments_count += segment_count

            # 获取数据集总episodes数
            total_episodes_count = current_data_loader.info.total_episodes

            return jsonify({
                "stats": stats,
                "global_stats": {
                    "annotated_episodes_count": annotated_episodes_count,
                    "total_episodes_count": total_episodes_count,
                    "total_segments_count": total_segments_count
                }
            })

        except Exception as e:
            logger.error(f"Error getting annotations stats: {e}")
            return jsonify({
                "stats": {},
                "global_stats": {
                    "annotated_episodes_count": 0,
                    "total_episodes_count": 0,
                    "total_segments_count": 0
                },
                "error": str(e)
            }), 500

    @app.route("/api/annotations/<int:episode_id>")
    def api_get_annotations(episode_id: int):
        """API: 获取episode的标注数据"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            annotation_data = current_annotation_store.load(episode_id)

            if annotation_data is None:
                # 返回空结构
                return jsonify({
                    "episode_id": episode_id,
                    "label_info": {"action_config": []}
                })

            return jsonify(annotation_data)

        except Exception as e:
            logger.error(f"Error getting annotations for episode {episode_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/annotations/<int:episode_id>", methods=["POST"])
    def api_save_annotations(episode_id: int):
        """API: 保存episode的标注数据"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"ok": False, "error": "No data provided"}), 400

            # 验证episode_id一致性
            if data.get("episode_id") != episode_id:
                return jsonify({"ok": False, "error": "Episode ID mismatch"}), 400

            # 后端必填字段验证
            if not data.get("task_name") or not data.get("task_name").strip():
                return jsonify({"ok": False, "error": "task_name is required and cannot be empty"}), 400

            if not data.get("init_scene_text") or not data.get("init_scene_text").strip():
                return jsonify({"ok": False, "error": "init_scene_text is required and cannot be empty"}), 400

            # 验证至少有一个段落
            action_config = data.get("label_info", {}).get("action_config", [])
            if not action_config or len(action_config) == 0:
                return jsonify({"ok": False, "error": "At least one action segment is required"}), 400

            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            # 保存标注
            result = current_annotation_store.upsert(episode_id, data)
            return jsonify(result)

        except ValueError as e:
            logger.warning(f"Invalid annotation data for episode {episode_id}: {e}")
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error saving annotations for episode {episode_id}: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/next_episode/<int:episode_id>")
    def api_next_episode(episode_id: int):
        """API: 获取下一个episode ID"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)

            total_episodes = current_data_loader.info.total_episodes
            next_id = (episode_id + 1) % total_episodes  # 循环回到开始

            return jsonify({"next_id": next_id})

        except Exception as e:
            logger.error(f"Error getting next episode for {episode_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/video_keys")
    def api_video_keys():
        """API: 获取数据集中所有视频特征键"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_data_loader = LocalDatasetLoader(current_dataset_path)

            video_keys = current_data_loader.get_video_keys()
            return jsonify(video_keys)

        except Exception as e:
            logger.error(f"Error getting video keys: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/annotations/<int:episode_id>", methods=["DELETE"])
    def api_delete_annotation(episode_id: int):
        """API: 删除单条episode标注"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            result = current_annotation_store.delete_episode(episode_id)
            return jsonify(result)

        except Exception as e:
            logger.error(f"Error deleting annotation for episode {episode_id}: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/annotations", methods=["DELETE"])
    def api_delete_all_annotations():
        """API: 删除全部标注"""
        try:
            data = request.get_json()

            # 需要确认才能删除全部
            if not data or not data.get("confirm"):
                return jsonify({
                    "ok": False,
                    "error": "Confirmation required. Send {\"confirm\": true} to delete all annotations."
                }), 400

            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            result = current_annotation_store.delete_all()
            return jsonify(result)

        except Exception as e:
            logger.error(f"Error deleting all annotations: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/history/task_names")
    def get_historical_task_names():
        """获取历史任务名称列表"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            all_annotations = current_annotation_store.load_all()
            task_names = []

            for annotation in all_annotations.values():
                task_name = annotation.get('task_name')
                if task_name and task_name.strip() and task_name not in task_names:
                    task_names.append(task_name)

            # 按使用频率排序（最近使用的在前面）
            task_names.sort()
            return jsonify({"task_names": task_names})

        except Exception as e:
            logger.error(f"Error getting historical task names: {e}")
            return jsonify({"task_names": []}), 200

    @app.route("/api/history/init_scenes")
    def get_historical_init_scenes():
        """获取历史初始场景描述列表"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            all_annotations = current_annotation_store.load_all()
            init_scenes = []

            for annotation in all_annotations.values():
                init_scene = annotation.get('init_scene_text')
                if init_scene and init_scene.strip() and init_scene not in init_scenes:
                    init_scenes.append(init_scene)

            init_scenes.sort()
            return jsonify({"init_scenes": init_scenes})

        except Exception as e:
            logger.error(f"Error getting historical init scenes: {e}")
            return jsonify({"init_scenes": []}), 200

    @app.route("/api/history/actions")
    def get_historical_actions():
        """获取历史动作描述列表"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            all_annotations = current_annotation_store.load_all()
            actions = []

            for annotation in all_annotations.values():
                action_config = annotation.get('label_info', {}).get('action_config', [])
                for segment in action_config:
                    action_text = segment.get('action_text')
                    if action_text and action_text.strip() and action_text not in actions:
                        actions.append(action_text)

            actions.sort()
            return jsonify({"actions": actions})

        except Exception as e:
            logger.error(f"Error getting historical actions: {e}")
            return jsonify({"actions": []}), 200

    @app.route("/api/history/skills")
    def get_historical_skills():
        """获取历史技能类型列表"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            all_annotations = current_annotation_store.load_all()
            skills = set()

            # 获取预定义的技能列表
            default_skills = ["Pick", "Place", "Move", "Rotate", "Push", "Pull", "Grasp", "Release"]
            skills.update(default_skills)

            # 添加历史使用的技能
            for annotation in all_annotations.values():
                action_config = annotation.get('label_info', {}).get('action_config', [])
                for segment in action_config:
                    skill = segment.get('skill')
                    if skill and skill.strip():
                        skills.add(skill)

            skills_list = sorted(list(skills))
            return jsonify({"skills": skills_list})

        except Exception as e:
            logger.error(f"Error getting historical skills: {e}")
            # 返回默认技能列表
            default_skills = ["Pick", "Place", "Move", "Rotate", "Push", "Pull", "Grasp", "Release"]
            return jsonify({"skills": default_skills}), 200

    @app.route("/api/skills/<skill_name>/usage")
    def api_check_skill_usage(skill_name: str):
        """API: 检查技能使用情况"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            result = current_annotation_store.check_skill_usage(skill_name)
            return jsonify(result)

        except Exception as e:
            logger.error(f"Error checking skill usage for {skill_name}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/skills/<old_name>", methods=["PUT"])
    def api_rename_skill(old_name: str):
        """API: 重命名技能"""
        try:
            data = request.get_json()
            if not data or 'new_name' not in data:
                return jsonify({"ok": False, "error": "new_name is required"}), 400

            new_name = data['new_name']
            if not new_name or not new_name.strip():
                return jsonify({"ok": False, "error": "new_name cannot be empty"}), 400

            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            result = current_annotation_store.rename_skill_globally(old_name, new_name.strip())
            if result["ok"]:
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            logger.error(f"Error renaming skill from {old_name}: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/skills/<skill_name>", methods=["DELETE"])
    def api_delete_skill(skill_name: str):
        """API: 删除技能（仅限未使用的）"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            current_dataset_config = DatasetConfig(current_dataset_path)
            current_annotation_store = AnnotationStore(current_dataset_config)

            result = current_annotation_store.delete_skill_if_unused(skill_name)
            if result["ok"]:
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            logger.error(f"Error deleting skill {skill_name}: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    # ========== Episode删除功能相关路由 ==========

    @app.route("/api/episode/<int:episode_id>/delete_preview")
    def api_episode_delete_preview(episode_id: int):
        """API: 获取删除episode的预览信息"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            dataset_manager = DatasetManager(current_dataset_path)

            preview = dataset_manager.get_delete_preview(episode_id)
            return jsonify(preview)

        except Exception as e:
            logger.error(f"Error getting delete preview for episode {episode_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/dataset/backup", methods=["POST"])
    def api_create_backup():
        """API: 创建数据集备份"""
        try:
            current_dataset_path = session.get('dataset_path', dataset_path)
            dataset_manager = DatasetManager(current_dataset_path)

            result = dataset_manager.create_backup()
            return jsonify(result)

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/dataset/split_by_skill", methods=["POST"])
    def api_split_dataset_by_skill():
        """API: 按技能分割数据集"""
        try:
            data = request.get_json()
            if not data or 'output_root' not in data:
                return jsonify({"success": False, "error": "output_root is required"}), 400

            output_root = data['output_root']
            if not output_root.strip():
                return jsonify({"success": False, "error": "output_root cannot be empty"}), 400

            # 验证输出目录
            output_path = Path(output_root)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return jsonify({"success": False, "error": f"无法创建输出目录: {e}"}), 400

            current_dataset_path = session.get('dataset_path', dataset_path)
            dataset_manager = DatasetManager(current_dataset_path, app_config)

            result = dataset_manager.split_by_skill(output_root)

            if result["success"]:
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            logger.error(f"数据集分割API失败: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/episode/<int:episode_id>", methods=["DELETE"])
    def api_delete_episode(episode_id: int):
        """API: 删除指定episode"""
        try:
            # 获取请求参数
            create_backup = request.args.get('backup', 'true').lower() == 'true'

            current_dataset_path = session.get('dataset_path', dataset_path)
            dataset_manager = DatasetManager(current_dataset_path)

            # 执行删除
            result = dataset_manager.delete_episode(episode_id, create_backup)

            if result["success"]:
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            logger.error(f"Error deleting episode {episode_id}: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        """404错误处理"""
        return "Page not found", 404

    @app.errorhandler(500)
    def internal_error(error):
        """500错误处理"""
        logger.error(f"Internal server error: {error}")
        return "Internal server error", 500

    return app