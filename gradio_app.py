"""
泰山派 RK3566 OCR 服务 - Gradio Web UI (独立/兼容入口)

用法:
  python gradio_app.py                      # 独立启动 Web UI (端口 7861)
  python gradio_app.py --port 7861          # 自定义端口
  python gradio_app.py --api-base http://localhost:8002  # 指定后端 API 地址

提示:
  推荐直接使用 `python main.py`，单端口 (8002) 同时提供 API 与 Web UI 服务。
"""

import os
import sys
import argparse
import urllib.request
import urllib.error

# 禁用遥测（避免无外网环境网络阻塞）
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from loguru import logger
from config import settings
from webui import build_interface


def check_backend_health(api_base: str):
    """检测后端 API 服务连通性并输出提示"""
    url = f"{api_base.rstrip('/')}/health"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                logger.info(f"✅ 成功连接到后端 OCR API 服务: {api_base}")
                return True
    except Exception:
        pass

    logger.warning(
        f"⚠️ 未检测到后端 API 服务 ({api_base})\n"
        f"   💡 提示: 请确保已执行 `python main.py` 启动后端 API，\n"
        f"   或使用 `--api-base http://<目标IP>:8002` 指定正确的后端地址。"
    )
    return False


def main():
    parser = argparse.ArgumentParser(description="泰山派 RK3566 OCR 服务 - Gradio 界面")
    parser.add_argument("--host", type=str, default=settings.HOST, help="监听主机地址")
    parser.add_argument("--port", type=int, default=settings.WEBUI_PORT, help="Web UI 监听端口 (默认 7861)")
    parser.add_argument(
        "--api-base",
        type=str,
        default=f"http://localhost:{settings.PORT}",
        help="后端 OCR API 地址 (默认 http://localhost:8002)",
    )
    parser.add_argument("--share", action="store_true", default=False, help="创建 Gradio 公网共享链接")
    args = parser.parse_args()

    # 设置 API 地址环境变量供 webui 模块读取
    os.environ["OCR_API_BASE"] = args.api_base

    logger.info("=" * 55)
    logger.info(f"📸 泰山派 OCR 服务 - 独立 Web UI 模式")
    logger.info(f"   界面地址:   http://{args.host}:{args.port}")
    logger.info(f"   对接后端:   {args.api_base}")
    logger.info("=" * 55)

    # 检查后端服务状态
    check_backend_health(args.api_base)

    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
