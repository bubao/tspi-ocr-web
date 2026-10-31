"""
泰山派 RK3566 OCR 服务主入口

启动方式:
  python main.py                    # 启动 API + Web UI (共用端口 8002)
  python main.py --api-only         # 仅启动 API (端口 8002)
  python main.py --port 8002        # 指定端口
  uvicorn main:app --host 0.0.0.0 --port 8002

访问地址:
  Web UI:    http://<ip>:8002/gradio/
  API 文档:  http://<ip>:8002/docs
  健康检查:  http://<ip>:8002/health
"""

import os
import sys
import time
import argparse
from contextlib import asynccontextmanager

# 禁用遥测（避免无外网环境网络阻塞）
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from config import settings
from services.stats import tracker


# ============ 日志配置 ============
def _setup_logging():
    """配置结构化统一日志输出"""
    log_level = "DEBUG" if settings.DEBUG else "INFO"
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": log_level,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            }
        ]
    )


_setup_logging()


# ============ 生命周期管理 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"📸 {settings.APP_NAME} v{settings.APP_VERSION} 启动中...")
    logger.info(f"   运行平台: 泰山派 RK3566 (ARM64)")

    # 模型预热（消除首次请求冷启动耗时）
    if settings.PRELOAD_MODEL:
        logger.info("⏳ 正在检查并预热 OCR 模型 (优先 RKNN NPU，无则回退 ONNX CPU)...")
        try:
            from routes.ocr import get_ocr_service
            service = get_ocr_service()
            if service and service.is_loaded():
                logger.info(f"✅ OCR 引擎加载完成: {service.engine_name} | 后端: {service.backend_name}")
            else:
                logger.warning("⚠️ OCR 模型已加载但就绪状态异常")
        except Exception as e:
            logger.error(f"❌ OCR 模型预热失败: {e}")
            logger.warning("服务将继续启动，首次请求时将重新尝试加载")

    logger.info("✅ 服务初始化完成，开始接收请求")
    yield
    logger.info("🛑 服务正在关闭...")
    logger.info("✅ 服务已安全停止")


# ============ FastAPI 实例 ============
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="泰山派 RK3566 离线 OCR 文字识别服务 (PP-OCRv4 ONNX / NPU加速)",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 统计中间件 ============
class StatsMiddleware(BaseHTTPMiddleware):
    """自动追踪 API 请求的耗时和成功状态"""

    async def dispatch(self, request, call_next):
        start = time.time()
        try:
            response = await call_next(request)
        except Exception:
            raise

        duration = time.time() - start
        path = request.url.path
        if path.startswith("/api/v1") and not path.endswith(("/docs", "/redoc", "/openapi.json")):
            success = response.status_code < 400
            tracker.record(path, request.method, duration, success=success)
        return response


app.add_middleware(StatsMiddleware)

# ============ 注册 API 路由 ============
from routes import ocr, stats

app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["OCR 文字识别"])
app.include_router(stats.router, prefix="/api/v1/stats", tags=["调用统计"])


# ============ 基础系统接口 ============
@app.get("/health", tags=["系统监控"])
async def health():
    """服务健康检查接口"""
    from routes.ocr import _ocr_service

    model_loaded = _ocr_service is not None and _ocr_service.is_loaded()
    return {
        "status": "healthy" if model_loaded else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "engine": settings.OCR_ENGINE,
        "backend": settings.INFERENCE_BACKEND,
        "model_loaded": model_loaded,
    }


@app.get("/", include_in_schema=False)
async def root():
    """根路径重定向：默认跳转至 Web UI 界面"""
    target = "/gradio/" if settings.ENABLE_WEBUI else "/docs"
    return RedirectResponse(url=target)


@app.post("/api/v1/ocr/debug", tags=["调试诊断"])
async def debug_ocr_upload(file: bytes):
    """直接上传图片二进制字节进行测试 (供终端 curl 快速诊断)"""
    logger.debug(f"[DEBUG] 接收到测试图片数据: {len(file)} bytes")
    try:
        from routes.ocr import _decode_image_bytes, get_ocr_service

        t0 = time.time()
        image_np = _decode_image_bytes(file)
        service = get_ocr_service()
        result = service.recognize(image_np, language="ch", det=True, rec=True, cls=False)
        duration = round(time.time() - t0, 3)

        return {
            "success": True,
            "duration": duration,
            "image_shape": list(image_np.shape),
            "text": result.get("text", ""),
            "results_count": len(result.get("results", [])),
        }
    except Exception as e:
        import traceback

        logger.error(f"[DEBUG] 识别异常: {e}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# ============ 挂载 Gradio Web UI ============
_webui_mounted = False


def mount_webui_app(target_app: FastAPI):
    """将 Gradio Web UI 挂载到 FastAPI 的 /gradio 子路径"""
    global _webui_mounted
    if _webui_mounted:
        return target_app

    try:
        from gradio.routes import mount_gradio_app
        from webui import build_interface

        mounted_app = mount_gradio_app(
            target_app,
            build_interface(),
            path="/gradio",
        )
        _webui_mounted = True
        return mounted_app
    except Exception as e:
        logger.error(f"❌ 挂载 Gradio Web UI 失败: {e}")
        return target_app


# 默认情况下如果启用了 Web UI，则在导入时挂载，确保 uvicorn main:app 也能正常访问 Web UI
if settings.ENABLE_WEBUI and not os.environ.get("OCR_API_ONLY"):
    app = mount_webui_app(app)


# ============ 服务启动入口 ============
def main():
    parser = argparse.ArgumentParser(description="泰山派 RK3566 OCR 服务启动入口")
    parser.add_argument("--host", type=str, default=settings.HOST, help="监听主机地址")
    parser.add_argument("--port", type=int, default=settings.PORT, help="监听端口 (默认 8002)")
    parser.add_argument("--api-only", action="store_true", default=False, help="仅启动 API 服务，不挂载 Web UI")
    parser.add_argument("--no-preload", action="store_true", default=False, help="启动时不预热加载模型")
    parser.add_argument("--reload", action="store_true", default=False, help="启用代码热重载 (开发模式)")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (RK3566 建议为 1)")
    args = parser.parse_args()

    # 应用命令行参数覆盖
    settings.HOST = args.host
    settings.PORT = args.port
    if args.no_preload:
        settings.PRELOAD_MODEL = False

    server_app = app
    if args.api_only:
        settings.ENABLE_WEBUI = False
        logger.info(f"🚀 启动模式: 仅 API 服务 (端口 {settings.PORT})")
        logger.info(f"   API 文档: http://{settings.HOST}:{settings.PORT}/docs")
    else:
        # 确保 Web UI 已挂载
        server_app = mount_webui_app(app)
        logger.info(f"🚀 启动模式: API + Web UI (共用端口 {settings.PORT})")
        logger.info(f"   Web UI:   http://{settings.HOST}:{settings.PORT}/gradio/")
        logger.info(f"   API 文档: http://{settings.HOST}:{settings.PORT}/docs")

    uvicorn.run(
        server_app,
        host=settings.HOST,
        port=settings.PORT,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
