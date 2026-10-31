#!/usr/bin/env python3
"""
泰山派 RK3566: PP-OCR ONNX 转 NCNN (带 NEON FP16 优化) CLI 命令行工具
基于 utils.ncnn_converter 模块
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.ncnn_converter import NCNNConverter


def main():
    parser = argparse.ArgumentParser(description="泰山派 RK3566: ONNX -> NCNN 转换命令行工具")
    parser.add_argument(
        "--source-dir", "-s",
        type=str,
        default=str(PROJECT_DIR / "models" / "ocr" / "ppocrv4"),
        help="源 ONNX 模型目录 (默认: models/ocr/ppocrv4)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="目标 NCNN 模型输出目录 (默认与源目录相同)"
    )
    parser.add_argument(
        "--det-onnx",
        type=str,
        default=None,
        help="单独指定要转换的检测模型路径"
    )
    parser.add_argument(
        "--rec-onnx",
        type=str,
        default=None,
        help="单独指定要转换的识别模型路径"
    )
    parser.add_argument(
        "--cls-onnx",
        type=str,
        default=None,
        help="单独指定要转换的方向分类模型路径"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="禁用 FP16 半精度，使用纯 FP32 格式导出"
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=2,
        help="pnnx 图优化等级 (默认: 2)"
    )

    args = parser.parse_args()

    converter = NCNNConverter(fp16=not args.no_fp16, opt_level=args.opt_level)
    if not converter.is_available():
        print("❌ 未检测到可用的 pnnx 工具链，请确认环境或运行 (venv) pip install pnnx")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" 🚀 泰山派 RK3566: ONNX ➔ NCNN 模型转换器")
    print(f" 精度模式: {'FP16 半精度 (推荐)' if not args.no_fp16 else 'FP32 浮点'}")
    print(f" pnnx 位置: {converter.pnnx_bin}")
    print("=" * 60)

    out_dir = Path(args.output_dir) if args.output_dir else None

    # 单独转换指定模型
    if args.det_onnx or args.rec_onnx or args.cls_onnx:
        if args.det_onnx:
            converter.convert_ppocr_det(Path(args.det_onnx), out_dir)
        if args.rec_onnx:
            converter.convert_ppocr_rec(Path(args.rec_onnx), out_dir)
        if args.cls_onnx:
            converter.convert_ppocr_cls(Path(args.cls_onnx), out_dir)
    else:
        # 批量套件转换
        src_dir = Path(args.source_dir)
        if not src_dir.exists():
            print(f"❌ 源目录不存在: {src_dir}")
            sys.exit(1)
        print(f" 正在扫描并转换目录中的 PP-OCR 套件: {src_dir}")
        results = converter.convert_ppocr_suite(src_dir, out_dir)
        print("\n转换结果清单:")
        for k, v in results.items():
            if isinstance(v, dict):
                status = "✅ 成功" if v.get("success") else "❌ 失败"
                print(f"  [{k}] {status}: {v.get('bin')}")
            else:
                print(f"  [{k}] 附带文件: {v}")

    print("\n🎉 全部处理完毕！\n")


if __name__ == "__main__":
    main()
