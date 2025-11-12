#!/usr/bin/env python3
"""
LeRobot 可视化系统启动脚本
功能: 命令行参数处理、配置加载、Flask应用启动
python main.py --dataset-path /mnt/data/kouyouyi/dataset/bottle_handover --port 9090
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from core.config import AppConfig
from web.app import create_app


def init_logging(level=logging.INFO):
    """初始化日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LeRobot Data Visualizer')

    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to the dataset directory (e.g., /path/to/grasp_dataset)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Host address to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to bind to (default: 9090)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # 初始化日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    init_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # 加载配置
        config = AppConfig(args.config)

        # 命令行参数覆盖配置文件
        dataset_path = args.dataset_path or config.get_dataset_path()
        host = args.host or config.get_app_host()
        port = args.port or config.get_app_port()
        debug = args.debug or config.get_debug_mode()

        logger.info(f"Starting LeRobot Visualizer")
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        logger.info(f"Debug mode: {debug}")

        # 验证数据集路径
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            logger.error(f"Dataset directory does not exist: {dataset_path}")
            sys.exit(1)

        info_file = dataset_dir / "meta" / "info.json"
        if not info_file.exists():
            logger.error(f"Dataset info.json not found: {info_file}")
            logger.error("Please make sure the dataset path is correct and contains meta/info.json")
            sys.exit(1)

        # 创建Flask应用
        app = create_app(dataset_path, args.config)

        logger.info(f"Server starting at http://{host}:{port}")
        logger.info("Press Ctrl+C to stop the server")

        # 启动服务器
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()