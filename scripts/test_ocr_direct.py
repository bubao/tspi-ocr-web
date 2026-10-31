#!/usr/bin/env python3
"""
泰山派 OCR 直接测试脚本 (绕过 HTTP/Gradio)
用法:
  python test_ocr_direct.py                  # 创建合成图片测试
  python test_ocr_direct.py path/to/image.jpg  # 用真实图片测试

用途: 直接在 RK3566 板上验证 OCR 引擎是否正常.
"""
import sys
import os
import traceback


def dbg(msg):
    """立即输出到 stderr"""
    print(msg, file=sys.stderr, flush=True)


dbg("=" * 60)
dbg("泰山派 OCR 直接测试 (绕过 HTTP/Gradio)")
dbg("=" * 60)

# Step 1: 导入基本库
try:
    import time
    import numpy as np
    from pathlib import Path
    dbg("[OK] 基本库 (time, numpy, pathlib)")
except Exception as e:
    dbg(f"[FAIL] 基本库导入: {e}")
    sys.exit(1)

# Step 2: 导入 OpenCV
try:
    import cv2
    dbg(f"[OK] OpenCV 版本: {cv2.__version__}")
except Exception as e:
    dbg(f"[FAIL] OpenCV: {e}")
    sys.exit(1)

# Step 3: 导入配置
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import settings
    dbg(f"[OK] 配置: PORT={settings.PORT}, OCR_ENGINE={settings.OCR_ENGINE}")
except Exception as e:
    dbg(f"[FAIL] 配置导入: {e}")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Step 4: 导入 OCR 服务与自动探测
try:
    from routes.ocr import get_ocr_service
    from services.ocr_service import check_rknn_available, check_ncnn_available
    has_rknn, rknn_msg, rknn_paths = check_rknn_available()
    has_ncnn, ncnn_msg, ncnn_paths = check_ncnn_available()
    dbg(f"[OK] OCR 引擎模块导入成功 (RKNN: {has_rknn}, NCNN: {has_ncnn})")
except Exception as e:
    dbg(f"[FAIL] OCR 服务导入: {e}")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Step 5: 检查模型文件
try:
    if has_rknn:
        dbg(f"RKNN 模型已就绪: det={rknn_paths['det'].name}, rec={rknn_paths['rec'].name}")
    elif has_ncnn:
        dbg(f"NCNN 模型已就绪: {ncnn_paths['dir'].name} (det.bin, rec.bin, dict.txt)")
    else:
        model_dir = settings.OCR_MODEL_DIR
        dbg(f"ONNX 模型目录: {model_dir}")
        dbg(f"  det: {(model_dir / settings.DET_MODEL).exists()}, rec: {(model_dir / settings.REC_MODEL).exists()}")
except Exception as e:
    dbg(f"[WARN] 模型目录检查: {e}")

# Step 6: 检查字典
try:
    dict_file = (
        ncnn_paths.get("dict") if has_ncnn else (settings.OCR_MODEL_DIR / settings.DICT_FILE)
    )
    if dict_file and dict_file.exists():
        with open(dict_file, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\r\n") for l in f]
        dict_len = len(lines)
        first = lines[0] if lines else ""
        dbg(f"字典就绪 ({dict_file.name}): {dict_len} 字符, 首字符={repr(first)}")
    else:
        dbg("[WARN] 字典文件不存在!")
except Exception as e:
    dbg(f"[WARN] 字典检查: {e}")

# Step 7: 加载模型
dbg("加载并初始化 OCR 引擎...")
try:
    engine = get_ocr_service()
    dbg(f"[OK] 模型加载就绪: {engine.engine_name} | 后端: {engine.backend_name}")
except Exception as e:
    dbg(f"[FAIL] 模型加载: {e}")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Step 8: 准备测试图片
test_image_path = sys.argv[1] if len(sys.argv) > 1 else None
try:
    if test_image_path and Path(test_image_path).exists():
        dbg(f"加载测试图片: {test_image_path}")
        img = cv2.imread(test_image_path)
        if img is None:
            dbg("[FAIL] cv2.imread 返回 None")
            sys.exit(1)
        dbg(f"  图片尺寸: {img.shape}, dtype={img.dtype}")
    else:
        # 创建合成图片
        dbg("创建合成测试图片")
        img = np.ones((80, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "TEST2024", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        dbg(f"  合成图片尺寸: {img.shape}")
except Exception as e:
    dbg(f"[FAIL] 测试图片: {e}")
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Step 9: 运行 OCR
dbg("开始 OCR 识别...")
try:
    start = time.time()
    result = engine.recognize(img, language="ch", det=True, rec=True, cls=False)
    elapsed = time.time() - start
    dbg(f"[OK] OCR 完成 耗时 {elapsed:.2f}s")
    dbg(f"  文本: {repr(result.get('text', '')[:100])}")
    dbg(f"  识别数: {len(result.get('results', []))}")
    dbg(f"  图片尺寸: {result.get('image_size', {})}")
    dbg(f"  处理时间: {result.get('processing_time', 0):.3f}s")
except Exception as e:
    dbg(f"[FAIL] OCR 识别失败:")
    dbg(f"  异常类型: {type(e).__name__}")
    dbg(f"  异常消息: {e}")
    traceback.print_exc(file=sys.stderr)

# Step 10: 检查输出类型
try:
    results = result.get("results", [])
    dbg(f"检查输出类型:")
    for i, r in enumerate(results):
        text = r.get("text", "")
        conf = r.get("confidence", 0)
        text_type = type(text).__name__
        conf_type = type(conf).__name__
        dbg(f"  [{i}] text={repr(text)} (conf={conf:.2%}) | text_type={text_type}, conf_type={conf_type}")
        if isinstance(text, bytes):
            dbg(f"    [BUG] text 是 bytes, 不是 str!")
except Exception as e:
    dbg(f"[WARN] 类型检查: {e}")

dbg("=" * 60)
dbg("测试完成!")
dbg("=" * 60)
