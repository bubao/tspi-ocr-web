#!/usr/bin/env python3
"""ONNX -> RKNN conversion for PP-OCRv4 (RK3566 NPU)"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

try:
    from rknn.api import RKNN
except ImportError:
    print("=" * 60)
    print("ERROR: 未找到 rknn.api 模块！")
    print("说明: rknn-toolkit2 模型转换工具仅支持 x86_64 Linux PC / 容器环境。")
    print("开发板端 (ARM64) 仅支持推理运行，不支持直接运行此转换脚本。")
    print("请在 x86 PC 或 Mac Docker 中执行此脚本: pip install rknn-toolkit2")
    print("=" * 60)
    sys.exit(1)


def convert(onnx_path, rknn_path, target="rk3566", input_size_list=None):
    name = Path(onnx_path).stem
    print(f"Converting: {name} -> {rknn_path}")
    rknn = RKNN(verbose=True)
    rknn.config(target_platform=target, optimization_level=3)

    load_kwargs = {"model": onnx_path}
    if input_size_list:
        load_kwargs["input_size_list"] = input_size_list
        load_kwargs["inputs"] = ["x"]

    ret = rknn.load_onnx(**load_kwargs)
    if ret != 0:
        print(f"Load failed: {ret}")
        return False
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"Build failed: {ret}")
        return False
    os.makedirs(os.path.dirname(rknn_path), exist_ok=True)
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"Export failed: {ret}")
        return False
    print(f"{name} OK")
    rknn.release()
    return True


def main():
    p = argparse.ArgumentParser(description="ONNX->RKNN for PP-OCRv4 (PC/容器环境运行)")
    p.add_argument("--det-onnx", type=str)
    p.add_argument("--rec-onnx", type=str)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--target", default="rk3566")
    p.add_argument("--auto", action="store_true")
    a = p.parse_args()

    if not a.output_dir:
        a.output_dir = str(settings.OCR_MODEL_DIR.parent / "ppocrv4-rknn")

    results = []

    # 针对 RK3566 NPU 的动态维度 ONNX，配置推荐固定尺寸
    det_input_size = [[1, 3, 480, 480]]
    rec_input_size = [[1, 3, 48, 320]]

    if a.auto or (not a.det_onnx and not a.rec_onnx):
        det = settings.OCR_MODEL_DIR / settings.DET_MODEL
        rec = settings.OCR_MODEL_DIR / settings.REC_MODEL
        if det.exists():
            out = os.path.join(a.output_dir, settings.DET_MODEL.replace(".onnx", ".rknn"))
            results.append(("det", convert(str(det), out, a.target, det_input_size)))
        if rec.exists():
            out = os.path.join(a.output_dir, settings.REC_MODEL.replace(".onnx", ".rknn"))
            results.append(("rec", convert(str(rec), out, a.target, rec_input_size)))
    else:
        if a.det_onnx:
            out = os.path.join(a.output_dir, Path(a.det_onnx).stem + ".rknn")
            results.append(("det", convert(a.det_onnx, out, a.target, det_input_size)))
        if a.rec_onnx:
            out = os.path.join(a.output_dir, Path(a.rec_onnx).stem + ".rknn")
            results.append(("rec", convert(a.rec_onnx, out, a.target, rec_input_size)))

    print(f"Result: {sum(1 for _, ok in results if ok)}/{len(results)}")


if __name__ == "__main__":
    main()
