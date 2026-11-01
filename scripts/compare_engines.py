#!/usr/bin/env python3
"""
泰山派 RK3566: 多 OCR 引擎对比评测工具
在相同测试图片与运行环境下，横向对比 PP-OCRv5 (NCNN FP16) 与 Indra OCR (ONNX 4核) 的推理耗时、识别准确率与模型体积。

用法:
  python scripts/compare_engines.py                       # 默认评测当前目录的 test_chinese.png
  python scripts/compare_engines.py --image path/to/img   # 指定自定义图片
  python scripts/compare_engines.py --rounds 3            # 多轮评测取平均耗时
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import cv2
from routes.ocr import get_ocr_service


def get_model_size_mb(path: Path) -> float:
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def benchmark_engine(name: str, engine_key: str, img, rounds: int = 1):
    print(f"\n{'=' * 65}")
    print(f" 🔍 正在评测引擎: {name} (engine='{engine_key}')")
    print(f"{'=' * 65}")

    try:
        service = get_ocr_service(engine_key)
    except Exception as e:
        print(f"❌ 加载引擎失败: {e}")
        return None

    # 预热一次
    print(" ⏳ 预热推理中...")
    service.recognize(img)

    times = []
    last_res = None
    for r in range(rounds):
        t0 = time.perf_counter()
        res = service.recognize(img)
        t_cost = time.perf_counter() - t0
        times.append(t_cost)
        last_res = res

    avg_time = sum(times) / len(times)
    core_time = last_res.get("processing_time", avg_time)

    print(f" ⏱️  平均总耗时: {avg_time * 1000:.1f} ms (单轮核心算法测量: {core_time * 1000:.1f} ms)")
    print(f" 📝 识别文本行数: {len(last_res.get('results', []))} 行")
    print("-" * 65)
    for i, line in enumerate(last_res.get("results", [])):
        print(f"  [{i + 1:02d}] 置信度 {line['confidence']:.2f} | {line['text']}")
    print("-" * 65)

    return {
        "name": name,
        "engine": last_res.get("engine", name),
        "backend": last_res.get("backend", "CPU"),
        "avg_time_ms": avg_time * 1000,
        "core_time_ms": core_time * 1000,
        "line_count": len(last_res.get("results", [])),
        "full_text": last_res.get("text", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="泰山派 RK3566 OCR 引擎横向对比评测工具")
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=str(PROJECT_DIR / "test_chinese.png"),
        help="待评测的测试图片路径 (默认: test_chinese.png)"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=2,
        help="评测轮数 (默认: 2 轮取平均值)"
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ 找不到测试图片: {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ 图片解码失败: {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print("\n" + "#" * 65)
    print(" 🏁 泰山派 RK3566 多 OCR 引擎性能与准确率对比测试")
    print(f" 测试图片: {img_path.name} ({w}x{h} px)")
    print(f" 测试轮数: {args.rounds} 轮")
    print("#" * 65)

    # 1. 评测 PP-OCRv5 NCNN
    res_ncnn = benchmark_engine(
        "PP-OCRv5 NCNN (FP16 汇编极速)", "ncnn", img, rounds=args.rounds
    )

    # 2. 评测 Indra OCR
    res_indra = benchmark_engine(
        "Indra OCR (1M超轻量多语言)", "indra", img, rounds=args.rounds
    )

    # 3. 输出综合对比表格
    print("\n" + "=" * 65)
    print(" 📊 评测对比总结报告")
    print("=" * 65)
    print(f"{'对比项':<18} | {'PP-OCRv5 NCNN':<20} | {'Indra OCR':<20}")
    print("-" * 65)
    print(f"{'模型设计':<18} | {'百度 PP-OCR 系列':<20} | {'Datalab Indra (1M)':<20}")
    print(f"{'模型总大小':<18} | {'~11 MB (det 2.5M+rec 8.5M)':<20} | {'~10 MB (det 390K+rec 9.7M)':<20}")
    print(f"{'检测模型大小':<18} | {'2.5 MB':<20} | {'389 KB (极小)':<20}")
    print(f"{'推理后端':<18} | {'NCNN (NEON FP16)':<20} | {'ONNX Runtime (4核 CPU)':<20}")

    if res_ncnn and res_indra:
        ncnn_time_str = f"{res_ncnn['avg_time_ms']:.1f} ms"
        indra_time_str = f"{res_indra['avg_time_ms']:.1f} ms"
        ncnn_lines_str = f"{res_ncnn['line_count']} 行"
        indra_lines_str = f"{res_indra['line_count']} 行"
        print(f"{'整图总耗时':<18} | {ncnn_time_str:<20} | {indra_time_str:<20}")
        print(f"{'识别有效行数':<18} | {ncnn_lines_str:<20} | {indra_lines_str:<20}")

    print("=" * 65)
    print("\n💡 建议选择指南:")
    print("  • 【中文自然场景 / 截屏票据】: 优先使用 PP-OCRv5 NCNN，中文词汇针对性强、标点和断句精准。")
    print("  • 【多语言 / PDF复杂文档 / 极致轻量】: 优先使用 Indra OCR，检测网络仅 390KB，多语言字符库丰富。\n")


if __name__ == "__main__":
    main()
