"""
RGA (Rockchip Raster Graphic Accelerator) 工具模块
基于 Rockchip 官方 librga (im2d + DRM DMA-BUF / PRIME Buffer) 连续物理显存加速
适配 Linux 6.1 内核，支持任意奇偶尺寸的高速硬件缩放，不可用时自动安全回退 OpenCV。
"""

import os
import ctypes
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

# ============ RGA 常量定义 (参考 /usr/include/rga/rga.h) ============
RK_FORMAT_RGBA_8888 = 0x0 << 8
RK_FORMAT_RGBX_8888 = 0x1 << 8
RK_FORMAT_RGB_888 = 0x2 << 8
RK_FORMAT_BGRA_8888 = 0x3 << 8
RK_FORMAT_BGR_888 = 0x7 << 8

# ============ 全局状态与动态库加载 ============
_rga_helper = None
_available = None

HELPER_SO_PATH = Path(__file__).resolve().parent / "librga_helper.so"


def _load_rga_helper():
    """加载经过编译的 RGA DRM DMA-BUF 辅助库"""
    global _rga_helper
    if _rga_helper is not None:
        return _rga_helper

    if not HELPER_SO_PATH.exists():
        # 如果动态库未编译，尝试就地使用系统 gcc 编译
        c_src = HELPER_SO_PATH.with_suffix(".c")
        if c_src.exists():
            cmd = f"gcc -shared -fPIC -O3 '{c_src}' -lrga -o '{HELPER_SO_PATH}' 2>/dev/null"
            os.system(cmd)

    if HELPER_SO_PATH.exists():
        try:
            lib = ctypes.CDLL(str(HELPER_SO_PATH))
            lib.rga_helper_init.restype = ctypes.c_int
            lib.rga_helper_deinit.restype = None
            lib.rga_helper_is_available.restype = ctypes.c_int
            lib.rga_helper_resize.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.rga_helper_resize.restype = ctypes.c_int
            _rga_helper = lib
            return _rga_helper
        except Exception as e:
            logger.debug(f"加载 librga_helper.so 异常: {e}")
            return None
    return None


def is_available() -> bool:
    """检测 RGA 硬件加速是否可用（带全局缓存）"""
    global _available
    if _available is not None:
        return _available

    # 1. 检测设备节点
    if not os.path.exists("/dev/rga"):
        _available = False
        return False

    # 2. 检查当前用户是否有读写权限
    if not os.access("/dev/rga", os.R_OK | os.W_OK):
        _available = False
        logger.info("ℹ️ 当前用户无 /dev/rga 读写权限，自动使用 OpenCV 进行图像缩放 (可运行 sudo chmod 666 /dev/rga 免 sudo 开启硬件加速)")
        return False

    # 3. 加载 DRM DMA-BUF 辅助驱动
    helper = _load_rga_helper()
    if helper is None:
        _available = False
        logger.info("ℹ️ 未能加载 librga_helper 扩展，图像缩放由 OpenCV 处理")
        return False

    try:
        ret = helper.rga_helper_init()
        if ret == 0:
            _available = True
            logger.info("✅ RGA 2D 硬件加速引擎已就绪 (DRM DMA-BUF 物理显存通道)")
            return True
        else:
            _available = False
            return False
    except Exception as e:
        _available = False
        logger.debug(f"RGA 初始化异常: {e}")
        return False


def rga_resize(src: np.ndarray, dst_width: int, dst_height: int, src_format: int = None) -> Optional[np.ndarray]:
    """
    使用 RGA 硬件芯片执行 2D 图像缩放 (基于 DRM DMA-BUF 连续显存)
    硬件无法处理或尺寸过小时安全返回 None，由调用方 fallback 到 OpenCV。
    """
    if not is_available():
        return None

    h, w = src.shape[:2]
    # RGA2 硬件对低于 16px 的极小尺寸不适用，直接由 CPU 极速处理
    if w < 16 or h < 16 or dst_width < 16 or dst_height < 16:
        return None

    # RGA2 硬件对大于 2048px 的超大图物理显存开销极大，直接由 CPU OpenCV 处理，防止耗尽内核 CMA 内存池导致 OOM
    if w > 2048 or h > 2048 or dst_width > 2048 or dst_height > 2048:
        return None

    # RGA2 硬件单次缩放限制: 缩放倍率在 1/16 ~ 16 之间 (0.0625 <= scale <= 16.0)
    # 超过 16 倍的极长/极宽文字切片直接回退 OpenCV，避免触发 librga scale 异常
    if w > dst_width * 16 or h > dst_height * 16 or dst_width > w * 16 or dst_height > h * 16:
        return None

    channels = 1 if len(src.shape) == 2 else src.shape[2]
    if channels not in (3, 4):
        return None

    helper = _load_rga_helper()
    if helper is None:
        return None

    try:
        if not src.flags["C_CONTIGUOUS"]:
            src = np.ascontiguousarray(src)

        dst = np.empty((dst_height, dst_width, channels), dtype=np.uint8)

        ret = helper.rga_helper_resize(
            src.ctypes.data_as(ctypes.c_char_p),
            w,
            h,
            dst.ctypes.data_as(ctypes.c_char_p),
            dst_width,
            dst_height,
            channels,
        )

        if ret == 0:
            return dst
        return None

    except Exception as e:
        logger.debug(f"RGA 硬件缩放异常: {e}")
        return None


def rga_resize_or_fallback(
    src: np.ndarray,
    dst_width: int,
    dst_height: int,
    src_format: int = None,
    enable_rga: bool = True,
) -> np.ndarray:
    """
    图像缩放：优先尝试 RGA 2D 硬件加速，不可用或失败时自动安全 fallback 到 OpenCV
    """
    import cv2

    if enable_rga:
        try:
            res = rga_resize(src, dst_width, dst_height, src_format)
            if res is not None:
                return res
        except Exception:
            pass

    return cv2.resize(src, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)


def rga_crop_or_fallback(src: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """图像裁剪 (内存视图切片)"""
    return src[y : y + h, x : x + w]
