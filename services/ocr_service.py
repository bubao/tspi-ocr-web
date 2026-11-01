"""
OCR 文字识别服务实现
支持引擎:
  - 智能自动切换 (优先 RKNN NPU 硬件加速，未检测到模型则回退 ONNX CPU 多线程)
  - PP-OCRv4 ONNX (CPU 4核并行，轻量且完全离线)
  - PP-OCRv4 RKNN (泰山派 RK3566 NPU 0.8 TOPS 硬件加速)
  - PaddleOCR (原生完整框架)
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from loguru import logger

from config import settings
from services.rga_utils import rga_resize_or_fallback


class BaseOCR(ABC):
    """OCR 引擎基类"""

    def __init__(self):
        self.engine = None
        self._loaded = False
        self.engine_name = "BaseOCR"
        self.backend_name = "CPU"

    @abstractmethod
    def load_model(self):
        """加载 OCR 模型"""
        pass

    @abstractmethod
    def recognize(
        self,
        image_input,
        language: str = "ch",
        det: bool = True,
        rec: bool = True,
        cls: bool = True,
    ) -> dict:
        """识别图片中的文字 (支持文件路径或内存 numpy 数组)"""
        pass

    def is_loaded(self) -> bool:
        return self._loaded

    def release(self):
        """释放模型与硬件资源"""
        pass


def check_rknn_available() -> Tuple[bool, str, Dict[str, Path]]:
    """
    检查系统中是否存在可用的 RKNN 模型与 RKNNLite 运行环境
    返回: (available: bool, reason: str, model_paths: dict)
    """
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        return False, "未安装 rknn-toolkit-lite2 运行库", {}

    # 扫描候选模型目录
    candidate_dirs = [
        settings.OCR_MODEL_DIR_RKNN,
        settings.OCR_MODEL_DIR,
        settings.MODEL_DIR / "ocr",
    ]

    det_names = [settings.RKNN_DET_MODEL, "ppocrv4_det.rknn", "det.rknn"]
    rec_names = [settings.RKNN_REC_MODEL, "ppocrv4_rec.rknn", "rec.rknn"]

    det_path = None
    rec_path = None

    for d in candidate_dirs:
        if not d.exists():
            continue
        if det_path is None:
            for name in det_names:
                p = d / name
                if p.exists() and p.stat().st_size > 10000:
                    det_path = p
                    break
        if rec_path is None:
            for name in rec_names:
                p = d / name
                if p.exists() and p.stat().st_size > 10000:
                    rec_path = p
                    break

    if not det_path or not rec_path:
        missing = []
        if not det_path:
            missing.append("det.rknn")
        if not rec_path:
            missing.append("rec.rknn")
        return False, f"未检测到完整 RKNN 模型文件 (缺少: {', '.join(missing)})", {}

    return True, "RKNN 模型与 NPU 运行环境已就绪", {"det": det_path, "rec": rec_path}


def check_ncnn_available() -> Tuple[bool, str, Dict[str, Path]]:
    """
    检查系统中是否存在可用的 NCNN 模型与 ncnn 运行库
    返回: (available: bool, reason: str, model_paths: dict)
    """
    try:
        import ncnn
    except ImportError:
        return False, "未安装 ncnn 运行库 (可通过 pip install ncnn 安装)", {}

    candidate_dirs = [
        settings.OCR_MODEL_DIR_NCNN,
        settings.OCR_MODEL_DIR,
    ]

    for d in candidate_dirs:
        if not d.exists():
            continue
        det_param = d / "det.param"
        det_bin = d / "det.bin"
        rec_param = d / "rec.param"
        rec_bin = d / "rec.bin"
        dict_txt = d / "dict.txt"

        if (
            det_param.exists()
            and det_bin.exists()
            and rec_param.exists()
            and rec_bin.exists()
            and dict_txt.exists()
        ):
            return True, "NCNN 模型与运行环境已就绪", {
                "dir": d,
                "det_param": det_param,
                "det_bin": det_bin,
                "rec_param": rec_param,
                "rec_bin": rec_bin,
                "dict": dict_txt,
                "cls_param": d / "cls.param",
                "cls_bin": d / "cls.bin",
            }

    return False, "未检测到完整 NCNN 模型文件 (缺少 det/rec param/bin)", {}


def check_indra_available() -> Tuple[bool, str, Dict[str, Path]]:
    """
    检查系统中是否存在可用的 Indra OCR 模型与 onnxruntime
    返回: (available: bool, reason: str, model_paths: dict)
    """
    try:
        import onnxruntime
    except ImportError:
        return False, "未安装 onnxruntime 运行库", {}

    indra_dir = Path("models/ocr/indra")
    if not indra_dir.exists():
        indra_dir = settings.OCR_MODEL_DIR.parent / "indra"

    det_path = indra_dir / "det.onnx"
    rec_cjk_path = indra_dir / "rec_cjk.onnx"
    rec_path = indra_dir / "rec.onnx"
    meta_path = indra_dir / "meta.json"

    if not det_path.exists():
        return False, f"未检测到 Indra 检测模型: {det_path.name}", {}
    if not (rec_cjk_path.exists() or rec_path.exists()):
        return False, "未检测到 Indra 识别模型 (rec_cjk.onnx 或 rec.onnx)", {}
    if not meta_path.exists():
        return False, f"未检测到 Indra 字典配置: {meta_path.name}", {}

    return True, "Indra OCR 模型与 ONNX 运行环境已就绪", {
        "dir": indra_dir,
        "det": det_path,
        "rec_cjk": rec_cjk_path if rec_cjk_path.exists() else None,
        "rec": rec_path if rec_path.exists() else None,
        "meta": meta_path,
    }


class PPOCRv4BaseEngine(BaseOCR):
    """PP-OCRv4 通用算法逻辑基类 (共享图像前处理、DBNet 轮廓后处理与 CTC 字典解码)"""

    def __init__(self):
        super().__init__()
        self.char_dict: List[str] = []
        self.rec_input_height = 48

    def _load_dict(self, dict_path: Path):
        """加载字符字典，严格保留第一行 \\u3000 全角空格以防字符映射错位"""
        if dict_path.exists():
            with open(dict_path, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n") for line in f]
            if lines and lines[0] == "blank":
                self.char_dict = lines
            else:
                self.char_dict = ["blank"] + lines

            for i, ch in enumerate(self.char_dict):
                if isinstance(ch, bytes):
                    self.char_dict[i] = ch.decode("utf-8", errors="replace")
            logger.info(f"   字典: {dict_path.name} ({len(self.char_dict)} 字符)")
        else:
            logger.warning(f"字典文件不存在: {dict_path}")
            self.char_dict = ["blank"] + [chr(i) for i in range(32, 127)]

    @abstractmethod
    def _run_det(self, blob: np.ndarray) -> np.ndarray:
        """执行检测模型推理"""
        pass

    @abstractmethod
    def _run_rec(self, blob: np.ndarray) -> np.ndarray:
        """执行识别模型推理"""
        pass

    def _run_cls(self, blob: np.ndarray) -> Optional[np.ndarray]:
        """执行方向分类器推理"""
        return None

    def recognize(
        self,
        image_input,
        language: str = "ch",
        det: bool = True,
        rec: bool = True,
        cls: bool = False,
        use_rga: bool = True,
    ) -> dict:
        """执行通用 OCR 文字识别"""
        if not self._loaded:
            self.load_model()

        start_time = time.perf_counter()

        # 图像格式标准化为 BGR uint8
        if isinstance(image_input, np.ndarray):
            img = image_input
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = img
        elif isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"无法读取图片文件: {image_input}")
        else:
            raise TypeError(f"不支持的图片输入类型: {type(image_input)}")

        h, w = img_bgr.shape[:2]
        image_size = {"width": w, "height": h}

        # 超过限制尺寸时等比缩小，防止爆显存与系统 OOM
        max_size = settings.MAX_IMAGE_SIZE
        scale_ratio = 1.0
        if max(w, h) > max_size:
            scale_ratio = max_size / max(w, h)
            new_w, new_h = int(w * scale_ratio), int(h * scale_ratio)
            img_bgr = rga_resize_or_fallback(img_bgr, new_w, new_h, enable_rga=use_rga)

        results = []

        if det:
            # 1. 文本检测 (内部自动限制长边为 960 进行低显存极速推理)
            dt_boxes = self._text_detect(img_bgr, use_rga=use_rga)

            # 2. 文本识别
            if rec and dt_boxes:
                for box in dt_boxes:
                    crop = self._crop_text_region(img_bgr, box)
                    if crop.size == 0:
                        continue

                    if cls and settings.USE_ANGLE_CLS:
                        crop = self._correct_orientation(crop)

                    text, confidence = self._text_recognize(crop, use_rga=use_rga)
                    if text.strip():
                        # 若发生过初始等比缩放，坐标无损映射回原图物理尺寸
                        orig_box = box / scale_ratio if scale_ratio != 1.0 else box
                        results.append(
                            {
                                "bbox": [float(p) for point in orig_box for p in point],
                                "text": text,
                                "confidence": float(confidence),
                            }
                        )
        elif rec:
            # 整图识别 (无检测阶段)
            text, confidence = self._text_recognize(img_bgr, use_rga=use_rga)
            if text.strip():
                results.append(
                    {
                        "bbox": [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)],
                        "text": text,
                        "confidence": float(confidence),
                    }
                )

        processing_time = round(time.perf_counter() - start_time, 4)
        full_text = "\n".join(r["text"] for r in results)

        return {
            "text": full_text,
            "results": results,
            "processing_time": processing_time,
            "image_size": image_size,
            "engine": self.engine_name,
            "backend": self.backend_name,
        }

    def _text_detect(self, img: np.ndarray, use_rga: bool = True) -> list:
        """文本检测 -> 返回检测框四边形列表"""
        h, w = img.shape[:2]

        # 严格遵守 PaddleOCR 官方标准：将检测输入长边限制为 DET_LIMIT_SIDE_LEN (默认 960)
        # 既能防止高分辨率大图在检测多层 FPN 反卷积与特征图展开时引发内存溢出 (OOM)，又能提速 10 倍以上
        limit_side = settings.DET_LIMIT_SIDE_LEN
        ratio = min(1.0, limit_side / max(h, w))
        new_h = max(int(round(h * ratio / 32) * 32), 32)
        new_w = max(int(round(w * ratio / 32) * 32), 32)

        resized = rga_resize_or_fallback(img, new_w, new_h, enable_rga=use_rga)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
        del resized

        pred = self._run_det(blob)
        del blob

        # 概率图二值化
        bitmap = (pred[0, 0] > 0.3).astype(np.uint8)
        del pred

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        del bitmap

        dt_boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            area = rect[1][0] * rect[1][1]
            if area < 100:
                continue

            # 无损映射回当前输入图 (img) 的真实物理坐标
            box[:, 0] *= w / new_w
            box[:, 1] *= h / new_h

            dt_boxes.append(self._sort_box(box))

        return dt_boxes

    def _crop_text_region(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """根据检测框裁剪文字区域 (带 DBNet 膨胀外扩，避免文字被截断)"""
        x_min = box[:, 0].min()
        x_max = box[:, 0].max()
        y_min = box[:, 1].min()
        y_max = box[:, 1].max()

        box_w = x_max - x_min
        box_h = y_max - y_min

        pad_x = box_w * 0.15
        pad_y = box_h * 0.35

        x_min_pad = int(max(0, x_min - pad_x))
        x_max_pad = int(min(img.shape[1], x_max + pad_x))
        y_min_pad = int(max(0, y_min - pad_y))
        y_max_pad = int(min(img.shape[0], y_max + pad_y))

        cropped = img[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
        if cropped.size == 0:
            return cropped

        # PaddleOCR 官方标准: 若为竖排文本 (高宽比 >= 1.5)，逆时针旋转 90 度转为横排供识别模型处理
        crop_h, crop_w = cropped.shape[:2]
        if crop_h >= crop_w * 1.5:
            cropped = np.rot90(cropped, 1)

        return cropped

    def _correct_orientation(self, img: np.ndarray) -> np.ndarray:
        """文字角度校正"""
        blob = cv2.resize(img, (192, 48)).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        pred = self._run_cls(blob)
        if pred is not None and pred[0][1] > 0.9:
            img = cv2.rotate(img, cv2.ROTATE_180)
        return img

    def _text_recognize(self, img: np.ndarray, use_rga: bool = True) -> tuple:
        """文字识别 -> 返回 (文字内容, 置信度)"""
        h, w = img.shape[:2]
        new_h = self.rec_input_height
        new_w = max(int(w * new_h / h), new_h)

        resized = rga_resize_or_fallback(img, new_w, new_h, enable_rga=use_rga)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        pred = self._run_rec(blob)
        text, confidence = self._ctc_decode(pred[0])
        return text, confidence

    def _ctc_decode(self, pred: np.ndarray) -> tuple:
        """CTC 贪心解码"""
        pred_idx = np.argmax(pred, axis=-1)
        chars = []
        prev = -1
        indices = pred_idx.flatten()

        confidences = []
        for t, idx in enumerate(indices):
            if idx != prev and idx != 0 and idx < len(self.char_dict):
                ch = self.char_dict[idx]
                if isinstance(ch, bytes):
                    try:
                        ch = ch.decode("utf-8", errors="replace")
                    except Exception:
                        ch = ""
                elif not isinstance(ch, str):
                    ch = str(ch)
                chars.append(ch)
                confidences.append(float(pred[t, idx]))
            prev = idx

        text = "".join(chars).strip()
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return text, avg_conf

    def _sort_box(self, box: np.ndarray) -> np.ndarray:
        """四边形顶点排序 (左上、右上、右下、左下)"""
        pts = box.reshape(4, 2)
        idx_by_y = np.argsort(pts[:, 1])
        top = pts[idx_by_y[:2]]
        bottom = pts[idx_by_y[2:]]

        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


class PPOCRv4ONNXEngine(PPOCRv4BaseEngine):
    """PP-OCRv4 ONNX 引擎 (基于 ONNXRuntime，CPU 4核并行)"""

    def __init__(self, model_dir: Optional[Path] = None):
        super().__init__()
        self.engine_name = "PP-OCRv4 (ONNX)"
        self.backend_name = "CPU (ONNXRuntime)"
        self.model_dir = model_dir or settings.OCR_MODEL_DIR
        self.det_session = None
        self.rec_session = None
        self.cls_session = None

    def load_model(self):
        """加载 ONNX 模型文件"""
        try:
            import onnxruntime as ort

            det_path = self.model_dir / settings.DET_MODEL
            rec_path = self.model_dir / settings.REC_MODEL
            dict_path = self.model_dir / settings.DICT_FILE

            if not det_path.exists():
                raise FileNotFoundError(f"检测模型不存在: {det_path}，请先下载模型")
            if not rec_path.exists():
                raise FileNotFoundError(f"识别模型不存在: {rec_path}，请先下载模型")

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = settings.ONNX_NUM_THREADS
            sess_options.inter_op_num_threads = settings.ONNX_INTER_THREADS
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            provider = "CPUExecutionProvider"

            self.det_session = ort.InferenceSession(str(det_path), sess_options=sess_options, providers=[provider])
            self.rec_session = ort.InferenceSession(str(rec_path), sess_options=sess_options, providers=[provider])

            # 动态检查输入高度
            rec_in_shape = self.rec_session.get_inputs()[0].shape
            if len(rec_in_shape) >= 3 and isinstance(rec_in_shape[2], int):
                self.rec_input_height = rec_in_shape[2]
            else:
                self.rec_input_height = 48

            cls_path = self.model_dir / settings.CLS_MODEL
            if cls_path.exists() and settings.USE_ANGLE_CLS:
                self.cls_session = ort.InferenceSession(str(cls_path), sess_options=sess_options, providers=[provider])

            self._load_dict(dict_path)
            self._loaded = True
            logger.info("✅ PP-OCRv4 ONNX 模型加载完成")
            logger.info(f"   检测模型: {det_path.name} | 识别模型: {rec_path.name}")
            logger.info(f"   推理后端: {provider} (线程: intra={settings.ONNX_NUM_THREADS}, inter={settings.ONNX_INTER_THREADS})")

        except Exception as e:
            logger.error(f"ONNX 模型加载失败: {e}")
            raise

    def _run_det(self, blob: np.ndarray) -> np.ndarray:
        input_name = self.det_session.get_inputs()[0].name
        return self.det_session.run(None, {input_name: blob.astype(np.float32)})[0]

    def _run_rec(self, blob: np.ndarray) -> np.ndarray:
        input_name = self.rec_session.get_inputs()[0].name
        return self.rec_session.run(None, {input_name: blob.astype(np.float32)})[0]

    def _run_cls(self, blob: np.ndarray) -> Optional[np.ndarray]:
        if self.cls_session is not None:
            input_name = self.cls_session.get_inputs()[0].name
            return self.cls_session.run(None, {input_name: blob.astype(np.float32)})[0]
        return None


class PPOCRv4RKNNEngine(PPOCRv4BaseEngine):
    """PP-OCRv4 RKNN 引擎 (基于 RKNNLite，泰山派 RK3566 NPU 硬件加速)"""

    def __init__(self, det_path: Optional[Path] = None, rec_path: Optional[Path] = None):
        super().__init__()
        self.engine_name = "PP-OCRv4 (RKNN)"
        self.backend_name = "NPU (RKNNLite 0.8TOPS)"
        self.det_path = det_path or settings.rknn_det_model_path
        self.rec_path = rec_path or settings.rknn_rec_model_path
        self.det_rknn = None
        self.rec_rknn = None

    def load_model(self):
        """加载 RKNN 模型至 NPU 运行时"""
        try:
            from rknnlite.api import RKNNLite

            if not self.det_path.exists():
                raise FileNotFoundError(f"RKNN 检测模型不存在: {self.det_path}")
            if not self.rec_path.exists():
                raise FileNotFoundError(f"RKNN 识别模型不存在: {self.rec_path}")

            logger.info("⏳ 正在初始化 RKNN NPU 运行时...")

            # 1. 初始化检测模型
            self.det_rknn = RKNNLite()
            ret = self.det_rknn.load_rknn(str(self.det_path))
            if ret != 0:
                raise RuntimeError(f"加载 RKNN 检测模型失败, ret={ret}")
            ret = self.det_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError(f"初始化 RKNN 检测运行时失败, ret={ret}")

            # 2. 初始化识别模型
            self.rec_rknn = RKNNLite()
            ret = self.rec_rknn.load_rknn(str(self.rec_path))
            if ret != 0:
                raise RuntimeError(f"加载 RKNN 识别模型失败, ret={ret}")
            ret = self.rec_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError(f"初始化 RKNN 识别运行时失败, ret={ret}")

            dict_path = settings.dict_file_path
            self._load_dict(dict_path)
            self._loaded = True
            logger.info("✅ PP-OCRv4 RKNN 模型加载完成，NPU 加速已开启")
            logger.info(f"   检测模型: {self.det_path.name}")
            logger.info(f"   识别模型: {self.rec_path.name}")

        except Exception as e:
            logger.error(f"RKNN 模型加载失败: {e}")
            raise

    def _run_det(self, blob: np.ndarray) -> np.ndarray:
        outputs = self.det_rknn.inference(inputs=[blob.astype(np.float32)])
        return outputs[0]

    def _run_rec(self, blob: np.ndarray) -> np.ndarray:
        outputs = self.rec_rknn.inference(inputs=[blob.astype(np.float32)])
        return outputs[0]

    def release(self):
        if self.det_rknn:
            self.det_rknn.release()
            self.det_rknn = None
        if self.rec_rknn:
            self.rec_rknn.release()
            self.rec_rknn = None
        self._loaded = False


class PPOCRv5NCNNEngine(PPOCRv4BaseEngine):
    """PP-OCRv5 NCNN 引擎 (基于 Tencent ncnn，ARM NEON FP16 汇编极速推理)"""

    def __init__(self, model_dir: Optional[Path] = None):
        super().__init__()
        self.engine_name = "PP-OCRv5 (NCNN)"
        self.backend_name = "CPU (NCNN NEON FP16)"
        self.model_dir = model_dir or settings.OCR_MODEL_DIR_NCNN
        self.det_net = None
        self.rec_net = None
        self.cls_net = None
        self.num_threads = settings.ONNX_NUM_THREADS
        self.det_in_name = "input"
        self.det_out_name = "output"
        self.rec_in_name = "input"
        self.rec_out_name = "output"
        self.cls_in_name = "input"
        self.cls_out_name = "output"

    def load_model(self):
        """加载 NCNN 模型参数与权重"""
        try:
            import ncnn

            det_param = self.model_dir / "det.param"
            det_bin = self.model_dir / "det.bin"
            rec_param = self.model_dir / "rec.param"
            rec_bin = self.model_dir / "rec.bin"
            cls_param = self.model_dir / "cls.param"
            cls_bin = self.model_dir / "cls.bin"
            dict_path = self.model_dir / "dict.txt"

            if not det_param.exists() or not det_bin.exists():
                raise FileNotFoundError(f"缺少 NCNN 检测模型: {det_param} 或 {det_bin}")
            if not rec_param.exists() or not rec_bin.exists():
                raise FileNotFoundError(f"缺少 NCNN 识别模型: {rec_param} 或 {rec_bin}")

            # 屏蔽模型加载期间冗余的算子废弃警告
            stderr_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)

            try:
                # 1. 检测网络
                self.det_net = ncnn.Net()
                self.det_net.opt.num_threads = self.num_threads
                self.det_net.load_param(str(det_param))
                self.det_net.load_model(str(det_bin))
                det_blobs = {b.name for b in self.det_net.blobs()}
                self.det_in_name = "input" if "input" in det_blobs else "in0"
                self.det_out_name = "output" if "output" in det_blobs else "out0"

                # 2. 识别网络 (开启 FP16 packed/storage/arithmetic 加速)
                self.rec_net = ncnn.Net()
                self.rec_net.opt.num_threads = self.num_threads
                self.rec_net.opt.use_fp16_packed = True
                self.rec_net.opt.use_fp16_storage = True
                self.rec_net.opt.use_fp16_arithmetic = True
                self.rec_net.load_param(str(rec_param))
                self.rec_net.load_model(str(rec_bin))
                rec_blobs = {b.name for b in self.rec_net.blobs()}
                self.rec_in_name = "input" if "input" in rec_blobs else "in0"
                self.rec_out_name = "output" if "output" in rec_blobs else "out0"

                # 3. 方向分类网络 (按需)
                if cls_param.exists() and cls_bin.exists() and settings.USE_ANGLE_CLS:
                    self.cls_net = ncnn.Net()
                    self.cls_net.opt.num_threads = self.num_threads
                    self.cls_net.load_param(str(cls_param))
                    self.cls_net.load_model(str(cls_bin))
                    cls_blobs = {b.name for b in self.cls_net.blobs()}
                    self.cls_in_name = "input" if "input" in cls_blobs else "in0"
                    self.cls_out_name = "output" if "output" in cls_blobs else "out0"

            finally:
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)

            # 4. 加载字符字典 (PP-OCRv5 keys txt 保持行对应)
            if dict_path.exists():
                with open(dict_path, "r", encoding="utf-8") as f:
                    self.char_dict = [l.rstrip("\r\n") for l in f]
                logger.info(f"   字典: {dict_path.name} ({len(self.char_dict)} 字符)")
            else:
                self.char_dict = ["blank"] + [chr(i) for i in range(32, 127)]

            self._loaded = True
            logger.info("✅ PP-OCRv5 NCNN 模型加载完成 (ARM NEON FP16 加速)")
            logger.info(f"   检测模型: {det_param.name} | 识别模型: {rec_param.name}")
            logger.info(f"   推理后端: NCNN (线程: {self.num_threads}, FP16=开启)")

        except Exception as e:
            logger.error(f"NCNN 模型加载失败: {e}")
            raise

    def _run_det(self, blob: np.ndarray) -> np.ndarray:
        import ncnn
        data = np.ascontiguousarray(blob[0], dtype=np.float32)
        in_mat = ncnn.Mat(data)
        stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            ex = self.det_net.create_extractor()
            ex.input(self.det_in_name, in_mat)
            ret, out_mat = ex.extract(self.det_out_name)
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
        return np.array(out_mat)[np.newaxis, ...]

    def _run_rec(self, blob: np.ndarray) -> np.ndarray:
        import ncnn
        data = np.ascontiguousarray(blob[0], dtype=np.float32)
        in_mat = ncnn.Mat(data)
        stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            ex = self.rec_net.create_extractor()
            ex.input(self.rec_in_name, in_mat)
            ret, out_mat = ex.extract(self.rec_out_name)
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
        return np.array(out_mat)[np.newaxis, ...]

    def _run_cls(self, blob: np.ndarray) -> Optional[np.ndarray]:
        if self.cls_net is None:
            return None
        import ncnn
        data = np.ascontiguousarray(blob[0], dtype=np.float32)
        in_mat = ncnn.Mat(data)
        stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            ex = self.cls_net.create_extractor()
            ex.input(self.cls_in_name, in_mat)
            ret, out_mat = ex.extract(self.cls_out_name)
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
        return np.array(out_mat)[np.newaxis, ...]

    def release(self):
        if self.det_net:
            self.det_net.clear()
            self.det_net = None
        if self.rec_net:
            self.rec_net.clear()
            self.rec_net = None
        if self.cls_net:
            self.cls_net.clear()
            self.cls_net = None
        self._loaded = False


class PaddleOCREngine(BaseOCR):
    """PaddleOCR 框架引擎 (可选)"""

    def __init__(self):
        super().__init__()
        self.model_dir = settings.OCR_MODEL_DIR
        self.engine_name = "PaddleOCR"
        self.backend_name = "PaddlePaddle"

    def load_model(self):
        try:
            from paddleocr import PaddleOCR

            self.engine = PaddleOCR(
                use_angle_cls=settings.USE_ANGLE_CLS,
                lang=settings.OCR_LANG,
                use_gpu=False,
                enable_mkldnn=True,
                cpu_threads=settings.ONNX_NUM_THREADS,
            )
            self._loaded = True
            logger.info("✅ PaddleOCR 模型加载成功")
        except Exception as e:
            logger.error(f"PaddleOCR 加载失败: {e}")
            raise

    def recognize(
        self,
        image_input,
        language: str = "ch",
        det: bool = True,
        rec: bool = True,
        cls: bool = True,
    ) -> dict:
        if not self._loaded:
            self.load_model()

        start_time = time.perf_counter()
        result = self.engine.ocr(image_input, det=det, rec=rec, cls=cls)

        results = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                confidence = line[1][1]
                results.append(
                    {
                        "bbox": [float(p) for point in box for p in point],
                        "text": str(text),
                        "confidence": float(confidence),
                    }
                )

        processing_time = round(time.perf_counter() - start_time, 4)
        full_text = "\n".join(r["text"] for r in results)

        return {
            "text": full_text,
            "results": results,
            "processing_time": processing_time,
            "image_size": {},
            "engine": self.engine_name,
            "backend": self.backend_name,
        }


class IndraOCREngine(BaseOCR):
    """
    Indra OCR 引擎 (基于 ONNX Runtime 4核 CPU 并行)
    特点:
      - 极致轻量 (检测模型仅 ~390KB，总模型 ~10MB)
      - 支持中文 (CJK)、英文及 80+ 种全球语言
      - 纯 Python + ONNXRuntime，支持 RGA 2D 硬件缩放加速
    """

    def __init__(self, model_dir: Optional[Path] = None):
        super().__init__()
        self.model_dir = Path(model_dir) if model_dir else (Path("models/ocr/indra"))
        self.engine_name = "Indra-OCR"
        self.backend_name = "ONNX Runtime (CPU 4核)"
        self.det_sess = None
        self.rec_sess = None
        self.vocab_cjk: List[str] = []
        self.vocab_latin: List[str] = []
        self.det_long_side = 1024

    def load_model(self):
        try:
            import json
            import onnxruntime as ort

            det_path = self.model_dir / "det.onnx"
            rec_cjk_path = self.model_dir / "rec_cjk.onnx"
            rec_path = self.model_dir / "rec.onnx"
            meta_path = self.model_dir / "meta.json"

            if not det_path.exists():
                raise FileNotFoundError(f"Indra 检测模型不存在: {det_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Indra 元数据配置不存在: {meta_path}")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = getattr(settings, "ONNX_NUM_THREADS", 4)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            logger.info(f"⏳ 正在加载 Indra OCR 模型 (目录: {self.model_dir})...")
            self.det_sess = ort.InferenceSession(str(det_path), opts, providers=["CPUExecutionProvider"])

            rec_target = rec_cjk_path if rec_cjk_path.exists() else rec_path
            if not rec_target.exists():
                raise FileNotFoundError(f"未找到可用 Indra 识别模型: {rec_cjk_path} 或 {rec_path}")

            self.rec_sess = ort.InferenceSession(str(rec_target), opts, providers=["CPUExecutionProvider"])

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "vocab_cjk" in meta:
                self.vocab_cjk = json.loads(meta["vocab_cjk"]).get("chars", [])
            if "vocab" in meta:
                self.vocab_latin = json.loads(meta["vocab"]).get("chars", [])

            self.det_long_side = int(meta.get("det_long_side", 1024))
            self._loaded = True
            logger.info(
                f"✅ Indra OCR 模型加载成功 (检测: {det_path.name}, 识别: {rec_target.name}, 字典: {len(self.vocab_cjk)} CJK字符)"
            )
        except Exception as e:
            logger.error(f"Indra OCR 模型加载失败: {e}")
            raise

    def recognize(
        self,
        image_input,
        language: str = "ch",
        det: bool = True,
        rec: bool = True,
        cls: bool = False,
        use_rga: bool = True,
    ) -> dict:
        if not self._loaded:
            self.load_model()

        start_time = time.perf_counter()

        # 归一化输入图像为 BGR numpy ndarray
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
        elif isinstance(image_input, dict):
            if "path" in image_input and image_input["path"]:
                img = cv2.imread(str(image_input["path"]))
            elif "array" in image_input and image_input["array"] is not None:
                img = np.array(image_input["array"])
            else:
                img = None
        elif hasattr(image_input, "convert"):  # PIL.Image
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            img = np.array(image_input)

        if img is None or img.size == 0:
            return {
                "text": "",
                "results": [],
                "processing_time": 0.0,
                "image_size": {},
                "engine": self.engine_name,
                "backend": self.backend_name,
            }

        if len(img.shape) == 2:
            gray = img
            h, w = gray.shape
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            h, w = gray.shape
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

        boxes = []
        if det:
            # 检测前处理: 长边缩放到 1024, pad 到 32 倍数, 转为 ink map (1.0 - gray/255.0)
            scale = self.det_long_side / max(h, w)
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            if use_rga:
                resized_gray = rga_resize_or_fallback(gray, nw, nh)
            else:
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                resized_gray = cv2.resize(gray, (nw, nh), interpolation=interp)

            ph = ((nh + 31) // 32) * 32
            pw = ((nw + 31) // 32) * 32
            ink = np.zeros((ph, pw), dtype=np.float32)
            ink[:nh, :nw] = 1.0 - (resized_gray.astype(np.float32) / 255.0)

            # 检测推理
            out = self.det_sess.run(None, {"x": ink[None, None]})[0][0]

            # 检测后处理: 解码连通域与上下边距距离
            prob = 1.0 / (1.0 + np.exp(-out[0]))
            mask = (prob > 0.5).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            if n > 1:
                s = 2
                flat = lab.ravel()
                keep = flat > 0
                idx = flat[keep]
                ys = (np.indices(lab.shape)[0].ravel()[keep].astype(np.float32) + 0.5) * s
                top = ys - out[1].ravel()[keep]
                bot = ys + out[2].ravel()[keep]
                cnt = np.bincount(idx, minlength=n).astype(np.float32)
                y0_arr = np.bincount(idx, weights=top, minlength=n) / np.maximum(cnt, 1)
                y1_arr = np.bincount(idx, weights=bot, minlength=n) / np.maximum(cnt, 1)

                for i in range(1, n):
                    if stats[i, cv2.CC_STAT_AREA] < 3:
                        continue
                    lh = max(1.0, y1_arr[i] - y0_arr[i])
                    ex = min(0.15 * lh, 0.5 * lh)
                    x0 = max(0.0, (stats[i, cv2.CC_STAT_LEFT] * s - ex) / scale)
                    x1 = min(float(w), ((stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) * s + ex) / scale)
                    y0 = max(0.0, float(y0_arr[i]) / scale)
                    y1 = min(float(h), float(y1_arr[i]) / scale)
                    if x1 > x0 and y1 > y0:
                        boxes.append([x0, y0, x1, y1])

                # 按自上而下阅读顺序排序
                boxes.sort(key=lambda b: (b[1], b[0]))
        else:
            boxes = [[0.0, 0.0, float(w), float(h)]]

        results = []
        if rec and boxes:
            vocab = self.vocab_cjk if self.vocab_cjk else self.vocab_latin
            for b in boxes:
                x0, y0, x1, y1 = b
                line_h = max(1.0, y1 - y0)
                # 按照 Indra 标准扩充留白
                iy0 = max(0, int(np.floor(y0 - 0.2 * line_h)))
                iy1 = min(h, int(np.ceil(y1 + 0.2 * line_h)))
                ix0 = max(0, int(np.floor(x0 - 0.3 * line_h)))
                ix1 = min(w, int(np.ceil(x1 + 0.3 * line_h)))

                crop = gray[iy0:iy1, ix0:ix1]
                if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    continue

                ch, cw = crop.shape
                tw = max(8, int(round(cw * (32.0 / ch))))
                tw = ((tw + 7) // 8) * 8

                if use_rga:
                    resized_crop = rga_resize_or_fallback(crop, tw, 32)
                else:
                    resized_crop = cv2.resize(crop, (tw, 32))

                crop_ink = 1.0 - (resized_crop.astype(np.float32) / 255.0)
                crop_x = crop_ink[None, None, :, :]
                widths = np.array([tw], dtype=np.int64)

                res = self.rec_sess.run(["ids", "conf", "lengths"], {"x": crop_x, "widths": widths})
                ids, confs, lens = res[0][0], res[1][0], res[2][0]
                n_tok = int(lens)
                seq = ids[:n_tok]
                keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
                valid_ids = seq[keep]
                text = "".join([vocab[idx - 1] for idx in valid_ids if 0 < idx <= len(vocab)])
                conf = float(confs[:n_tok][keep].mean()) if keep.any() else 0.0

                if text.strip():
                    results.append(
                        {
                            "bbox": [
                                float(x0),
                                float(y0),
                                float(x1),
                                float(y0),
                                float(x1),
                                float(y1),
                                float(x0),
                                float(y1),
                            ],
                            "text": text,
                            "confidence": round(conf, 4),
                        }
                    )

        processing_time = round(time.perf_counter() - start_time, 4)
        full_text = "\n".join(r["text"] for r in results)

        return {
            "text": full_text,
            "results": results,
            "processing_time": processing_time,
            "image_size": {"width": w, "height": h},
            "engine": self.engine_name,
            "backend": self.backend_name,
        }

    def release(self):
        self.det_sess = None
        self.rec_sess = None
        self._loaded = False

