"""
ONNX -> NCNN 模型转换专用模块
基于腾讯 ncnn 官方 pnnx 工具，支持自动探测环境、提取独立工具链、动态尺寸定义及 FP16 精度压缩。
"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger


class NCNNConverter:
    """
    ONNX 转 NCNN 模型转换器
    支持任意通用 ONNX 模型的转换，并内置 PP-OCR 系列检测/识别/分类网络的特定动态尺寸预设。
    """

    def __init__(
        self,
        pnnx_path: Optional[Path] = None,
        fp16: bool = True,
        opt_level: int = 2,
    ):
        self.fp16 = fp16
        self.opt_level = opt_level
        self.pnnx_bin = self._find_or_extract_pnnx(pnnx_path)

    @staticmethod
    def _find_or_extract_pnnx(custom_path: Optional[Path] = None) -> Optional[Path]:
        """按优先级探测或无依赖提取 pnnx 独立可执行文件"""
        # 1. 显式指定的自定义路径
        if custom_path and Path(custom_path).exists() and os.access(str(custom_path), os.X_OK):
            return Path(custom_path)

        # 2. 项目内缓存路径
        project_root = Path(__file__).resolve().parent.parent
        project_pnnx = project_root / "tools" / "pnnx_bin" / "pnnx" / "pnnx"
        if project_pnnx.exists() and os.access(str(project_pnnx), os.X_OK):
            return project_pnnx

        # 3. 系统环境变量 PATH
        sys_pnnx = shutil.which("pnnx")
        if sys_pnnx:
            return Path(sys_pnnx)

        # 4. 自动从 pip wheel 缓存提取 (仅需 pnnx 自身独立 C++ 二进制，无需下载 500MB PyTorch)
        candidate_wheels = list(Path("/tmp").glob("**/pnnx*.whl")) + list(
            Path.home().glob(".cache/pip/**/pnnx*.whl")
        )
        if candidate_wheels:
            wheel_file = candidate_wheels[0]
            target_dir = project_root / "tools" / "pnnx_bin"
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(wheel_file, "r") as z:
                    z.extractall(target_dir)
                bin_path = target_dir / "pnnx" / "pnnx"
                if bin_path.exists():
                    bin_path.chmod(0o755)
                    logger.info(f"✅ 从 wheel 成功提取独立 pnnx 工具: {bin_path}")
                    return bin_path
            except Exception as e:
                logger.debug(f"解压 pnnx wheel 失败: {e}")

        return None

    def is_available(self) -> bool:
        """检查 pnnx 转换工具链是否就绪"""
        return self.pnnx_bin is not None and self.pnnx_bin.exists()

    def convert(
        self,
        onnx_path: Path,
        output_dir: Optional[Path] = None,
        param_name: Optional[str] = None,
        bin_name: Optional[str] = None,
        input_shapes: Optional[List[str]] = None,
        fp16: Optional[bool] = None,
    ) -> Tuple[bool, Optional[Path], Optional[Path]]:
        """
        执行通用 ONNX -> NCNN 转换

        :param onnx_path: 源 ONNX 模型路径
        :param output_dir: 输出目录 (默认与源文件同目录)
        :param param_name: 目标 .param 文件名
        :param bin_name: 目标 .bin 文件名
        :param input_shapes: 输入维度规格 (如 ["inputshape=[1,3,48,320]", "inputshape2=[1,3,48,640]"])
        :param fp16: 是否启用 FP16 半精度加速 (为 None 时使用实例默认值)
        :return: (success: bool, param_path: Path, bin_path: Path)
        """
        if not self.is_available():
            logger.error("❌ 未找到可用的 pnnx 可执行文件，无法执行转换")
            return False, None, None

        onnx_file = Path(onnx_path).resolve()
        if not onnx_file.exists():
            logger.error(f"❌ 输入 ONNX 文件不存在: {onnx_file}")
            return False, None, None

        out_dir = Path(output_dir).resolve() if output_dir else onnx_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = onnx_file.stem
        param_file = out_dir / (param_name or f"{stem}.param")
        bin_file = out_dir / (bin_name or f"{stem}.bin")

        use_fp16 = self.fp16 if fp16 is None else fp16
        shapes = input_shapes or []

        cmd = [
            str(self.pnnx_bin),
            str(onnx_file),
            *shapes,
            f"ncnnparam={param_file}",
            f"ncnnbin={bin_file}",
            f"fp16={1 if use_fp16 else 0}",
            f"optlevel={self.opt_level}",
        ]

        logger.info(f"🚀 开始转换: {onnx_file.name} ➔ {param_file.name} / {bin_file.name}")
        logger.debug(f"   命令: {' '.join(cmd)}")

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode != 0:
                logger.error(f"❌ pnnx 转换执行失败 (code {res.returncode}):\n{res.stderr or res.stdout}")
                return False, None, None

            # 清理转换产生的附属调试脚本和中间临时文件
            self._cleanup_intermediates(onnx_file.parent, stem)
            if out_dir != onnx_file.parent:
                self._cleanup_intermediates(out_dir, stem)

            if param_file.exists() and bin_file.exists():
                size_mb = bin_file.stat().st_size / 1024 / 1024
                logger.info(f"✅ 转换完成: {bin_file.name} ({size_mb:.2f}MB, FP16={use_fp16})")
                return True, param_file, bin_file
            else:
                logger.error("❌ 转换流程未报错但未找到目标 .param/.bin 文件")
                return False, None, None

        except subprocess.TimeoutExpired:
            logger.error("❌ 转换超时 (限制 600 秒)")
            return False, None, None
        except Exception as e:
            logger.error(f"❌ 转换异常: {e}")
            return False, None, None

    @staticmethod
    def _cleanup_intermediates(target_dir: Path, stem: str):
        """清理 pnnx 伴随生成的中间文件"""
        patterns = [
            f"{stem}.pnnx.*",
            f"{stem}*pnnx*.py",
            f"{stem}*ncnn*.py",
            f"{stem}*.pnnxsim.*",
        ]
        for pattern in patterns:
            for junk in target_dir.glob(pattern):
                try:
                    junk.unlink(missing_ok=True)
                except Exception:
                    pass

    # ============ 针对 PP-OCR 的专用转换预设 ============

    def convert_ppocr_det(
        self,
        onnx_path: Path,
        output_dir: Optional[Path] = None,
        param_name: str = "det.param",
        bin_name: str = "det.bin",
    ) -> Tuple[bool, Optional[Path], Optional[Path]]:
        """
        转换文本检测模型 (DBNet)
        预设动态分辨率: 宽高 32 对齐，支持动态图尺寸输入
        """
        shapes = ["inputshape=[1,3,96,416]", "inputshape2=[1,3,640,640]"]
        return self.convert(
            onnx_path=onnx_path,
            output_dir=output_dir,
            param_name=param_name,
            bin_name=bin_name,
            input_shapes=shapes,
            fp16=self.fp16,
        )

    def convert_ppocr_rec(
        self,
        onnx_path: Path,
        output_dir: Optional[Path] = None,
        param_name: str = "rec.param",
        bin_name: str = "rec.bin",
    ) -> Tuple[bool, Optional[Path], Optional[Path]]:
        """
        转换文本识别模型 (CRNN / SVTR / LCNet)
        预设高度固定 48px，宽度支持任意动态步长
        """
        shapes = ["inputshape=[1,3,48,320]", "inputshape2=[1,3,48,640]"]
        return self.convert(
            onnx_path=onnx_path,
            output_dir=output_dir,
            param_name=param_name,
            bin_name=bin_name,
            input_shapes=shapes,
            fp16=self.fp16,
        )

    def convert_ppocr_cls(
        self,
        onnx_path: Path,
        output_dir: Optional[Path] = None,
        param_name: str = "cls.param",
        bin_name: str = "cls.bin",
    ) -> Tuple[bool, Optional[Path], Optional[Path]]:
        """
        转换方向分类器模型
        预设标准输入形状: [1, 3, 80, 160] 或 [1, 3, 48, 192]
        """
        shapes = ["inputshape=[1,3,80,160]"]
        return self.convert(
            onnx_path=onnx_path,
            output_dir=output_dir,
            param_name=param_name,
            bin_name=bin_name,
            input_shapes=shapes,
            fp16=self.fp16,
        )

    def convert_ppocr_suite(
        self,
        source_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """批量转换一个目录下检测、识别与方向分类的完整模型套件"""
        src = Path(source_dir)
        dst = Path(output_dir) if output_dir else src
        results = {}

        det_onnx = src / "det.onnx"
        if det_onnx.exists():
            ok, p, b = self.convert_ppocr_det(det_onnx, dst)
            results["det"] = {"success": ok, "param": p, "bin": b}

        rec_onnx = src / "rec.onnx"
        if rec_onnx.exists():
            ok, p, b = self.convert_ppocr_rec(rec_onnx, dst)
            results["rec"] = {"success": ok, "param": p, "bin": b}

        cls_onnx = src / "cls.onnx"
        if cls_onnx.exists():
            ok, p, b = self.convert_ppocr_cls(cls_onnx, dst)
            results["cls"] = {"success": ok, "param": p, "bin": b}

        # 拷贝字典文件
        dict_src = src / "dict.txt"
        if dict_src.exists() and dst != src:
            shutil.copy2(dict_src, dst / "dict.txt")
            results["dict"] = dst / "dict.txt"

        return results
