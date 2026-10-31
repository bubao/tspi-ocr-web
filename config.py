"""
泰山派 RK3566 OCR 服务配置
支持: 智能自动选择 (优先 RKNN NPU，未检测到则无缝回退 ONNX CPU) / 手动指定
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 禁用 Gradio / HuggingFace 在无外网环境下的遥测阻塞
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class Settings(BaseSettings):
    # ============ 服务基础配置 ============
    APP_NAME: str = "泰山派 OCR 服务"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = False

    # ============ Web UI 配置 ============
    ENABLE_WEBUI: bool = True
    WEBUI_PORT: int = 7861

    # ============ 模型目录 ============
    MODEL_DIR: Path = Path(__file__).resolve().parent / "models"
    OCR_MODEL_DIR: Path = MODEL_DIR / "ocr" / "ppocrv4"
    OCR_MODEL_DIR_NCNN: Path = MODEL_DIR / "ocr" / "ppocrv5-ncnn"
    OCR_MODEL_DIR_RKNN: Path = MODEL_DIR / "ocr" / "ppocrv4-rknn"

    # ============ OCR 引擎与模式 ============
    # 可选: "auto" (智能探测: 优先 RKNN NPU，其次 NCNN FP16，无则回退 ONNX) | "rknn" | "ncnn" | "onnx" | "paddle"
    OCR_ENGINE: str = "auto"

    # 预加载模型 (启动时预热，消除首次推理冷启动耗时)
    PRELOAD_MODEL: bool = True

    # ============ 语言支持 ============
    OCR_LANG: str = "ch"

    # ============ 推理后端 (用于指示或覆盖) ============
    # 可选: "auto" | "rknn" | "cpu"
    INFERENCE_BACKEND: str = "auto"

    # ============ ONNX 线程配置 (RK3566 4核 Cortex-A55) ============
    ONNX_NUM_THREADS: int = 4
    ONNX_INTER_THREADS: int = 1

    # ============ 图片处理限制 ============
    MAX_IMAGE_SIZE: int = 2560  # 输入图片最大边长限制 (超大图等比缩小，防止爆显存与超时)
    DET_LIMIT_SIDE_LEN: int = 960  # 文本检测最大边长限制 (PaddleOCR 官方标准：加速 10~20 倍，坐标自动无损映射回原图)
    SUPPORTED_FORMATS: list[str] = ["jpg", "jpeg", "png", "bmp", "webp", "tiff"]

    # ============ ONNX 模型文件名 ============
    DET_MODEL: str = "det.onnx"
    REC_MODEL: str = "rec.onnx"
    CLS_MODEL: str = "cls.onnx"
    DICT_FILE: str = "dict.txt"

    # ============ RKNN 模型文件名 ============
    RKNN_DET_MODEL: str = "det.rknn"
    RKNN_REC_MODEL: str = "rec.rknn"

    # ============ NCNN 模型文件名 (PP-OCRv5) ============
    NCNN_DET_PARAM: str = "det.param"
    NCNN_DET_BIN: str = "det.bin"
    NCNN_REC_PARAM: str = "rec.param"
    NCNN_REC_BIN: str = "rec.bin"
    NCNN_CLS_PARAM: str = "cls.param"
    NCNN_CLS_BIN: str = "cls.bin"

    # ============ 性能与特征配置 ============
    USE_ANGLE_CLS: bool = False  # 是否启用角度分类器
    ENABLE_RGA: bool = True  # 是否尝试使用 RGA 2D 硬件加速缩放 (失败时自动安全回退 OpenCV)

    # ============ 跨域配置 ============
    CORS_ORIGINS: list[str] = ["*"]

    # 模型文件完整路径便捷属性
    @property
    def det_model_path(self) -> Path:
        return self.OCR_MODEL_DIR / self.DET_MODEL

    @property
    def rec_model_path(self) -> Path:
        return self.OCR_MODEL_DIR / self.REC_MODEL

    @property
    def cls_model_path(self) -> Path:
        return self.OCR_MODEL_DIR / self.CLS_MODEL

    @property
    def dict_file_path(self) -> Path:
        return self.OCR_MODEL_DIR / self.DICT_FILE

    @property
    def rknn_det_model_path(self) -> Path:
        # 优先在 ppocrv4-rknn 寻找，若无则检查 ppocrv4 目录
        p1 = self.OCR_MODEL_DIR_RKNN / self.RKNN_DET_MODEL
        if p1.exists():
            return p1
        return self.OCR_MODEL_DIR / self.RKNN_DET_MODEL

    @property
    def rknn_rec_model_path(self) -> Path:
        p1 = self.OCR_MODEL_DIR_RKNN / self.RKNN_REC_MODEL
        if p1.exists():
            return p1
        return self.OCR_MODEL_DIR / self.RKNN_REC_MODEL

    # ============ NCNN 便捷路径属性 ============
    @property
    def ncnn_det_param_path(self) -> Path:
        return self.OCR_MODEL_DIR_NCNN / self.NCNN_DET_PARAM

    @property
    def ncnn_det_bin_path(self) -> Path:
        return self.OCR_MODEL_DIR_NCNN / self.NCNN_DET_BIN

    @property
    def ncnn_rec_param_path(self) -> Path:
        return self.OCR_MODEL_DIR_NCNN / self.NCNN_REC_PARAM

    @property
    def ncnn_rec_bin_path(self) -> Path:
        return self.OCR_MODEL_DIR_NCNN / self.NCNN_REC_BIN

    @property
    def ncnn_cls_param_path(self) -> Path:
        return self.OCR_MODEL_DIR_NCNN / self.NCNN_CLS_PARAM

    @property
    def ncnn_cls_bin_path(self) -> Path:
        return self.OCR_MODEL_DIR_NCNN / self.NCNN_CLS_BIN

    @property
    def ncnn_dict_path(self) -> Path:
        return self.OCR_MODEL_DIR_NCNN / self.DICT_FILE

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
