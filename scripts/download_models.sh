#!/bin/bash
# ============ 泰山派 RK3566 OCR 服务模型下载脚本 ============
# 从 monkt/paddleocr-onnx (HuggingFace / ModelScope) 下载 ONNX 模型
#
# 推荐安装 ModelScope 走国内镜像:
#   (venv) pip install modelscope
#   python scripts/download_models.py
#
# 运行方式:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/download_models.sh  # 走镜像

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/ocr/ppocrv4}"

echo "==============================================="
echo " 泰山派 RK3566 OCR 模型下载"
echo "==============================================="
echo " 模型目录: $MODEL_DIR"
echo " 来源: monkt/paddleocr-onnx (HuggingFace)"
echo ""

mkdir -p "$MODEL_DIR"

# ============ 下载函数 ============
download_file() {
    local url="$1"
    local output="$2"
    local expected_size="${3:-0}"  # 期望字节数 (0=跳过校验)
    local tmp_output="${output}.tmp"
    
    if [ -f "$output" ]; then
        local local_size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo 0)
        if [ "$expected_size" -gt 0 ] && [ "$local_size" -ne "$expected_size" ]; then
            echo "  ⚠️  已存在但大小不符 (本地: $local_size, 期望: $expected_size)，重新下载"
            rm -f "$output"
        else
            echo "  ✅ 已存在: $(basename "$output") ($(numfmt --to=iec $local_size 2>/dev/null || echo ${local_size}B))"
            return 0
        fi
    fi
    
    echo "  ⬇️  下载: $(basename "$output")..."
    
    # 下载到临时文件
    if command -v wget &> /dev/null; then
        wget -4 -q --show-progress -O "$tmp_output" "$url" || {
            echo "  ❌ 下载失败: $url"
            rm -f "$tmp_output"
            return 1
        }
    elif command -v curl &> /dev/null; then
        curl -4 -L --progress-bar -o "$tmp_output" "$url" || {
            echo "  ❌ 下载失败: $url"
            rm -f "$tmp_output"
            return 1
        }
    else
        echo "  ❌ 需要 wget 或 curl"
        return 1
    fi
    
    # 校验文件大小
    local actual_size=$(stat -f%z "$tmp_output" 2>/dev/null || stat -c%s "$tmp_output" 2>/dev/null || echo 0)
    if [ "$expected_size" -gt 0 ] && [ "$actual_size" -ne "$expected_size" ]; then
        echo "  ❌ 文件损坏: 期望 $expected_size 字节，实际 $actual_size 字节"
        echo "     临时文件保留: $tmp_output"
        return 1
    fi
    
    # 通过校验，移动为正式文件
    mv "$tmp_output" "$output"
    echo "  ✅ 完成: $(basename "$output") ($(numfmt --to=iec $actual_size 2>/dev/null || echo ${actual_size}B))"
}

# ============ 模型下载 ============
echo ""
echo "📸 === PP-OCR ONNX 模型 (monkt/paddleocr-onnx) ==="
echo ""

BASE_URL="${HF_ENDPOINT:-https://hf-mirror.com}/monkt/paddleocr-onnx/resolve/main"

# 1. 文本检测模型 v3 (2.4MB = 2429873 bytes)
echo " --- 检测模型 (v3) ---"
download_file "$BASE_URL/detection/v3/det.onnx" "$MODEL_DIR/det.onnx" 2429873

# 2. 中文识别模型 (~84MB = 84468836 bytes)
echo " --- 识别模型 (chinese) ---"
download_file "$BASE_URL/languages/chinese/rec.onnx" "$MODEL_DIR/rec.onnx" 84468836

# 3. 文本方向分类模型 (~6.5MB = 6774157 bytes)
echo " --- 方向分类模型 (cls) ---"
download_file "$BASE_URL/preprocessing/textline-orientation/PP-LCNet_x1_0_textline_ori.onnx" "$MODEL_DIR/cls.onnx" 6774157

# 4. 中文字典文件 (74KB = 74012 bytes)
echo " --- 字典文件 ---"
download_file "$BASE_URL/languages/chinese/dict.txt" "$MODEL_DIR/dict.txt" 74012

# ============ 完成 ============
echo ""
echo "==============================================="
echo " ✅ OCR 模型下载完成 (含完整性校验)！"
echo ""
echo " 模型目录:"
echo "  $MODEL_DIR/"
echo " ├── det.onnx       (检测, 2.4MB)"
echo " ├── rec.onnx       (中文识别, ~84MB)"
echo " ├── cls.onnx       (方向分类, ~6.5MB)"
echo " └── dict.txt       (中文字典, ~74KB)"
echo ""
echo " 启动服务:"
echo "  pip install -r requirements.txt"
echo "  python main.py"
echo ""
echo " 如模型损坏:"
echo "  删除损坏文件重新运行脚本即可"
echo "==============================================="
