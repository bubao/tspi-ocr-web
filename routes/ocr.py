"""
OCR 文字识别 API
支持: PP-OCRv4, 通用文字/表格/票据/证件识别
只返回 YOLO 式标注数据 (bbox + text + confidence)，前端自行渲染
"""

import gc
import io
import time
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
import numpy as np
from loguru import logger

from config import settings

router = APIRouter()


class OCRResponse(BaseModel):
    success: bool
    text: str = ""
    results: list = []
    processing_time: float = 0.0
    image_size: dict = {}
    engine: Optional[str] = None
    backend: Optional[str] = None


class OCRRegion(BaseModel):
    bbox: list  # [x1, y1, x2, y2, x3, y3, x4, y4]
    text: str
    confidence: float


# ============ OCR 服务实例缓存（支持多引擎按需切换） ============
_engines_cache = {}


def get_ocr_service(engine: Optional[str] = None):
    global _engines_cache
    engine_target = (engine or settings.OCR_ENGINE or "auto").lower()

    if engine_target in _engines_cache:
        return _engines_cache[engine_target]

    if engine_target == "auto":
        from services.ocr_service import check_rknn_available, check_ncnn_available

        has_rknn, reason, rknn_paths = check_rknn_available()
        if has_rknn:
            logger.info(
                f"🚀 [自动检测] 发现 RKNN 模型 ({rknn_paths['det'].name}, {rknn_paths['rec'].name})，已自动启用 NPU 硬件加速推理"
            )
            from services.ocr_service import PPOCRv4RKNNEngine

            svc = PPOCRv4RKNNEngine(det_path=rknn_paths["det"], rec_path=rknn_paths["rec"])
        else:
            has_ncnn, ncnn_reason, ncnn_paths = check_ncnn_available()
            if has_ncnn:
                logger.info(f"⚡ [自动检测] {reason}，检测到 PP-OCRv5 NCNN 模型，已自动启用 NCNN (NEON FP16 极速) 引擎")
                from services.ocr_service import PPOCRv5NCNNEngine

                svc = PPOCRv5NCNNEngine(model_dir=ncnn_paths["dir"])
            else:
                logger.info(f"ℹ️ [自动检测] 未检测到 RKNN/NCNN 模型，自动选择使用 ONNX (CPU 4核并行) 引擎")
                from services.ocr_service import PPOCRv4ONNXEngine

                svc = PPOCRv4ONNXEngine()
    elif engine_target == "rknn":
        from services.ocr_service import PPOCRv4RKNNEngine

        svc = PPOCRv4RKNNEngine()
    elif engine_target == "ncnn":
        from services.ocr_service import PPOCRv5NCNNEngine

        svc = PPOCRv5NCNNEngine()
    elif engine_target == "onnx":
        from services.ocr_service import PPOCRv4ONNXEngine

        svc = PPOCRv4ONNXEngine()
    elif engine_target == "indra":
        from services.ocr_service import IndraOCREngine

        svc = IndraOCREngine()
    elif engine_target == "paddle":
        from services.ocr_service import PaddleOCREngine

        svc = PaddleOCREngine()
    else:
        raise ValueError(f"不支持的 OCR 引擎: {engine_target}")

    svc.load_model()
    _engines_cache[engine_target] = svc
    return svc


# ============ 工具函数 ============
def _decode_image_bytes(content: bytes) -> np.ndarray:
    """将上传的图片文件解码为 BGR numpy 数组 (内存中，不写盘)"""
    import cv2
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    del nparr
    if img is None:
        raise ValueError("图片解码失败，请确认文件为有效图片格式")

    # 保护机制: 若解码后为极端超大图片 (例如手机全景图 > 2560px)，在最外层即刻等比降采样，避免后续全链路内存溢出
    h, w = img.shape[:2]
    max_side = settings.MAX_IMAGE_SIZE
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


# ============ 通用文字识别 ============
@router.post("/recognize", response_model=OCRResponse)
async def recognize_image(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="语言 (ch/en/japan/korean)"),
    det: bool = Query(True, description="是否检测文本区域"),
    rec: bool = Query(True, description="是否识别文字"),
    cls: bool = Query(False, description="是否使用角度分类"),
    engine: Optional[str] = Query(None, description="指定推理引擎 (auto/onnx/rknn)"),
    use_rga: bool = Query(True, description="是否尝试使用 RGA 硬件加速缩放 (失败时自动回退 OpenCV)"),
):
    """
    上传图片进行 OCR 文字识别 (全流程内存处理，纯标注返回)

    - **file**: 图片文件 (jpg/png/bmp)
    - **language**: 中文(默认) / 英文 / 日文 / 韩文
    - **det**: 是否进行文本检测
    - **rec**: 是否进行文字识别
    - **cls**: 是否检测文字方向
    - **engine**: 指定模型引擎 (auto: 智能探测 / onnx: CPU 4核 / rknn: NPU加速)
    - **use_rga**: 是否尝试使用 RGA 硬件加速 (失败时自动回退 OpenCV)
    """
    start_time = time.time()

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片过大 (最大 20MB)")

    logger.debug(f"/recognize: file={file.filename}, size={len(content)} bytes, lang={language}, engine={engine}, use_rga={use_rga}")

    try:
        # 内存解码 → numpy (不落盘)
        image_np = _decode_image_bytes(content)
        del content  # 及时释放原始上传字节流
        service = get_ocr_service(engine=engine)
        lang = language or settings.OCR_LANG

        # 直接传入 numpy 数组，全流程内存处理
        result = service.recognize(image_np, language=lang, det=det, rec=rec, cls=cls, use_rga=use_rga)
        del image_np
        gc.collect()

        # 优先使用 service 内部精确测量的核心 OCR 处理耗时
        processing_time = result.get("processing_time") or round(time.perf_counter() - start_time, 4)

        return OCRResponse(
            success=True,
            text=result.get("text", ""),
            results=result.get("results", []),
            processing_time=processing_time,
            image_size=result.get("image_size", {}),
            engine=result.get("engine"),
            backend=result.get("backend"),
        )
    except Exception as e:
        logger.error(f"/recognize 识别异常: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


# ============ 简化识别（纯文本返回） ============
@router.post("/text")
async def extract_text(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="语言"),
):
    """
    上传图片，只返回识别的纯文本 (不落盘)
    """
    try:
        content = await file.read()

        image_np = _decode_image_bytes(content)
        del content

        service = get_ocr_service()
        lang = language or settings.OCR_LANG

        result = service.recognize(image_np, language=lang, det=True, rec=True, cls=False)
        del image_np
        gc.collect()

        return {
            "text": result.get("text", ""),
            "results": result.get("results", []),
            "image_size": result.get("image_size", {}),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"OCR 文本提取失败: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


# ============ URL 识别 ============
@router.post("/url")
async def recognize_url(
    url: str = Query(..., description="图片 URL"),
    language: Optional[str] = Query(None, description="语言"),
):
    """
    从 URL 下载图片并识别 (全流程内存处理)
    """
    import aiohttp

    start_time = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="无法下载图片")
                content = await resp.read()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"下载失败: {str(e)}")

    try:
        image_np = _decode_image_bytes(content)
        del content

        service = get_ocr_service()
        lang = language or settings.OCR_LANG

        result = service.recognize(image_np, language=lang, det=True, rec=True, cls=False)
        del image_np
        gc.collect()

        # 优先使用 service 内部精确测量的核心 OCR 处理耗时
        processing_time = result.get("processing_time") or round(time.perf_counter() - start_time, 4)

        return OCRResponse(
            success=True,
            text=result.get("text", ""),
            results=result.get("results", []),
            processing_time=processing_time,
            image_size=result.get("image_size", {}),
            engine=result.get("engine"),
            backend=result.get("backend"),
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"OCR URL 识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


# ============ 服务信息 ============
@router.get("/info", tags=["OCR 文字识别"])
async def get_engine_info():
    """获取 OCR 引擎与当前运行状态信息"""
    from services.rga_utils import is_available as rga_available

    service = get_ocr_service()
    return {
        "engine": getattr(service, "engine_name", settings.OCR_ENGINE),
        "backend": getattr(service, "backend_name", settings.INFERENCE_BACKEND),
        "language": settings.OCR_LANG,
        "model_dir": str(settings.OCR_MODEL_DIR),
        "model_loaded": service.is_loaded() if service else False,
        "supported_formats": settings.SUPPORTED_FORMATS,
        "rga_available": rga_available(),
    }
