#!/usr/bin/env python3
"""
泰山派 RK3566 OCR 模型下载脚本
支持下载:
  1. PP-OCRv5 (NCNN 高性能版, 强烈推荐, 仅 ~11MB, 推理约 150ms)
     来源: https://huggingface.co/qefro/pp-OCRv5-6-ncnn
  2. PP-OCRv4 (ONNX 原版, ~90MB, CPU 推理约 1.5s)
     来源: https://huggingface.co/monkt/paddleocr-onnx

支持参数:
  python scripts/download_models.py              # 默认检查并下载缺少的推荐模型 (优先 NCNN)
  python scripts/download_models.py --ncnn       # 仅下载/检查 PP-OCRv5 NCNN 模型
  python scripts/download_models.py --onnx       # 仅下载/检查 PP-OCRv4 ONNX 模型
  python scripts/download_models.py --all        # 下载全部模型 (NCNN + ONNX)
  python scripts/download_models.py --force      # 强制重新下载并覆盖
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============ 路径与配置 ============
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_DIR / "models" / "ocr"

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# 模型配置清单
MODEL_PACKS = {
    "ncnn": {
        "title": "PP-OCRv5 (NCNN 高性能版, 推荐)",
        "repo_id": "qefro/pp-OCRv5-6-ncnn",
        "dir": MODEL_ROOT / "ppocrv5-ncnn",
        "files": {
            "PP_OCRv5_mobile_det.param": "det.param",
            "PP_OCRv5_mobile_det.bin": "det.bin",
            "PP_OCRv5_mobile_rec.param": "rec.param",
            "PP_OCRv5_mobile_rec.bin": "rec.bin",
            "PP_LCNet_x0_25_textline_ori.param": "cls.param",
            "PP_LCNet_x0_25_textline_ori.bin": "cls.bin",
            "ppocr_keys_v5.txt": "dict.txt",
        },
    },
    "onnx": {
        "title": "PP-OCRv4 (ONNX 原版)",
        "repo_id": "monkt/paddleocr-onnx",
        "dir": MODEL_ROOT / "ppocrv4",
        "files": {
            "detection/v3/det.onnx": "det.onnx",
            "languages/chinese/rec.onnx": "rec.onnx",
            "preprocessing/textline-orientation/PP-LCNet_x1_0_textline_ori.onnx": "cls.onnx",
            "languages/chinese/dict.txt": "dict.txt",
        },
    },
}


def download_file_curl(url: str, output_path: Path) -> bool:
    """使用 curl 下载单个文件 (带断点续传与重试)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".incomplete")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        "curl", "-4", "-L", "--progress-bar",
        "--retry", "3", "--retry-delay", "2",
        "-C", "-",
        "-o", str(tmp_path),
        url,
    ]
    try:
        res = subprocess.run(cmd, timeout=300)
        if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 100:
            tmp_path.rename(output_path)
            return True
        if tmp_path.exists():
            tmp_path.unlink()
        return False
    except Exception as e:
        print(f"  ❌ 下载异常: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def download_pack(pack_key: str, force: bool = False) -> bool:
    """下载特定模型包"""
    pack = MODEL_PACKS[pack_key]
    dest_dir = pack["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f" 📦 正在检查/下载: {pack['title']}")
    print(f" 来源仓库: {pack['repo_id']}")
    print(f" 保存目录: {dest_dir}")
    print(f"{'='*55}")

    all_success = True
    base_url = f"{HF_ENDPOINT}/{pack['repo_id']}/resolve/main"

    for remote_path, local_name in pack["files"].items():
        local_path = dest_dir / local_name
        if not force and local_path.exists() and local_path.stat().st_size > 100:
            size_mb = local_path.stat().st_size / 1024 / 1024
            print(f"  ✅ 已就绪: {local_name:<16} ({size_mb:.2f}MB)")
            continue

        url = f"{base_url}/{remote_path}"
        print(f"  ⬇️  下载中: {local_name} (源自: {remote_path})")
        ok = download_file_curl(url, local_path)
        if ok:
            size_mb = local_path.stat().st_size / 1024 / 1024
            print(f"  ✅ 完成: {local_name:<16} ({size_mb:.2f}MB)")
        else:
            print(f"  ❌ 失败: {local_name}")
            all_success = False

    return all_success


def main():
    force = "--force" in sys.argv or "-f" in sys.argv
    download_all = "--all" in sys.argv
    only_onnx = "--onnx" in sys.argv
    only_ncnn = "--ncnn" in sys.argv

    print("\n" + "=" * 55)
    print("  📸 泰山派 RK3566 OCR 服务模型下载管理器")
    print("=" * 55)

    if download_all:
        targets = ["ncnn", "onnx"]
    elif only_onnx:
        targets = ["onnx"]
    elif only_ncnn:
        targets = ["ncnn"]
    else:
        # 默认模式: 优先下载推荐的高性能 NCNN 模型
        targets = ["ncnn"]

    overall_ok = True
    for t in targets:
        ok = download_pack(t, force=force)
        if not ok:
            overall_ok = False

    print("\n" + "=" * 55)
    if overall_ok:
        print(" 🎉 模型准备完毕！可以直接启动服务：")
        print("    python main.py")
    else:
        print(" ⚠️ 部分模型下载失败，请检查网络后重新运行。")
    print("=" * 55 + "\n")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
