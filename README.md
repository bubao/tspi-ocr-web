# 📸 泰山派 RK3566 离线 OCR 服务

专为 **泰山派 (RK3566, 4核 A55 + 2GB 内存)** 打造的高性能离线图片文字识别服务。

---

## 💡 架构设计说明（必读）

本项目内置**智能引擎自动侦测机制**，默认开箱即用：

```
                             ┌──> [检测到 RKNN 模型] ──> NPU 硬件加速推理 (0.8 TOPS RKNNLite, ~80ms)
                             │
 上传图片 ──> 启动自动检测 ──┼──> [检测到 NCNN 模型] ──> NCNN 极速引擎 (PP-OCRv5 + NEON FP16, ~150ms) ⭐推荐
                             │
                             └──> [回退 ONNX 引擎] ──> ONNX 引擎 (4核 CPU 并行 + RGA 硬件缩放, ~1.5s)
```

1. **智能多引擎自动分级切换（推荐）**：
   - 默认配置为 `OCR_ENGINE=auto`；
   - 服务启动时自动扫描模型目录：
     1. **优先检测 RKNN 模型**（`models/ocr/ppocrv4-rknn/`）：若存在 `det.rknn` 与 `rec.rknn`，优先启用板载 0.8 TOPS NPU 硬件加速；
     2. **其次检测 NCNN 模型**（`models/ocr/ppocrv5-ncnn/`）：若存在 PP-OCRv5 NCNN 模型，自动启用 ARM NEON FP16 汇编极速引擎，模型仅 11MB，推理耗时直接缩短至 **150ms 左右**；
     3. **兜底回退 ONNX 引擎**（`models/ocr/ppocrv4/`）：若以上均未找到，无缝平滑回退至通用 ONNX 引擎。
2. **PC 端转换 vs 板端推理分工说明**：
   - 瑞芯微官方模型转换工具（`rknn-toolkit2`）依赖 x86 交叉编译库，**仅可在 x86_64 Linux PC / 容器中运行**；
   - 泰山派板端仅运行推理库（`rknn-toolkit-lite2` 与 `ncnn`），负责加载模型进行高帧率低延迟推理。

---

## 📋 核心特性

- **三引擎智能切换**：优先 NPU 硬件加速（RKNN），秒级平滑分级（NCNN NEON FP16 -> ONNX）。
- **极速响应**：PP-OCRv5 NCNN 引擎整图识别约 **120~180ms** + RGA 2D 硬件缩放辅助。
- **超小内存占用**：PP-OCRv5 NCNN 模型总体量仅 **11MB**（识别模型仅 7.9MB），相比 ONNX（90MB）大幅节约板载内存。
- **完全离线**：本地私有化部署，不依赖外部网络连接。
- **零磁盘 IO**：图片全流程内存解码处理，不产生临时文件、不损耗闪存寿命。
- **纯前端标注渲染（零服务端画图开销）**：
  - 后端仅负责算法推理，API 仅返回包含 `bbox`（8点坐标）、`text`、`confidence` 的轻量 JSON 元数据；
  - Web UI 前端采用 `gr.AnnotatedImage` 动态渲染组件，直接利用客户端（手机/电脑）浏览器的 Canvas/DOM 渲染高亮线框与悬停标签；
  - 彻底省去了板端在 CPU 上使用 PIL 逐像素绘图、加载中文字体、二次编码大图的繁重计算与内存占用。
- **单端口部署**：FastAPI 后端与 Gradio Web UI 共用端口 `8002`，也可独立解耦运行。
- **一站式工作台**：左右分栏，支持拖拽、点击、Ctrl+V 粘贴截图与一键复制纯文本。

---

## 🚀 快速开始（开发板端直接运行）

### 1. 安装 Python 依赖

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装板端运行时依赖
pip install -r requirements.txt
```

> ⚠️ **注意**：板端请务必安装 `opencv-python-headless`，切勿安装桌面版 `opencv-python`，否则会因缺失桌面图形库报错 `libGL.so.1`。

### 2. 检查或下载模型

本项目支持自动拉取开箱即用的轻量模型（默认优先推荐并下载 **PP-OCRv5 NCNN 极速版**）：

```bash
# 自动检查并下载推荐的 PP-OCRv5 NCNN 极速模型 (~11MB)
python scripts/download_models.py

# 其它可选下载参数:
python scripts/download_models.py --ncnn       # 仅下载/更新 NCNN 模型
python scripts/download_models.py --onnx       # 仅下载 PP-OCRv4 ONNX 模型
python scripts/download_models.py --all        # 下载全部模型包 (NCNN + ONNX)
python scripts/download_models.py --force      # 强制重新拉取覆盖
```

#### 📦 开源模型仓库与资源索引
1. **[qefro/pp-OCRv5-6-ncnn (HuggingFace)](https://huggingface.co/qefro/pp-OCRv5-6-ncnn)**：
   - 本项目 NCNN 默认搭载的 PP-OCRv5 移动端精简模型；
   - 包含文本检测 (`det.bin` 2.3MB)、文本识别 (`rec.bin` 7.9MB) 与方向分类 (`cls.bin` 475KB)；
   - 经过 NEON FP16 深度压缩与优化，在泰山派 4 核 Cortex-A55 上实测仅需 **~150ms**。
2. **[assafvayner/translatte-ocr-ncnn (HuggingFace)](https://huggingface.co/assafvayner/translatte-ocr-ncnn)**：
   - 社区维护的高性能 translatte OCR NCNN 多语言模型仓库；
   - 提供检测网络与针对 CJK（中日韩）、拉丁、阿拉伯语等多语种独立的 NCNN 权重。
3. **[monkt/paddleocr-onnx (HuggingFace)](https://huggingface.co/monkt/paddleocr-onnx)**：
   - 原版 PP-OCRv4 ONNX 格式模型（~90MB），适合作为跨平台保底与对照。

### 3. 启动服务

```bash
# 推荐：单端口同时启动 API + Web UI (端口 8002)
python main.py

# 其它启动模式选项:
python main.py --api-only           # 仅启动 RESTful API
python main.py --port 8002          # 自定义端口
python gradio_app.py --port 7861     # 独立启动 Gradio 前端
```

### 4. 访问服务

- **Web UI 界面**: `http://<开发板IP>:8002/gradio/`
- **Swagger API 文档**: `http://<开发板IP>:8002/docs`
- **健康监控**: `http://<开发板IP>:8002/health`

---

## 📡 API 调用示例

### 1. 图片 OCR 识别（含坐标标注）

```bash
curl -X POST "http://localhost:8002/api/v1/ocr/recognize" \
  -F "file=@test.png" \
  -F "language=ch"
```

**返回示例**：
```json
{
  "success": true,
  "text": "泰山派 RK3566",
  "results": [
    {
      "bbox": [20.0, 30.0, 180.0, 30.0, 180.0, 65.0, 20.0, 65.0],
      "text": "泰山派 RK3566",
      "confidence": 0.985
    }
  ],
  "processing_time": 0.148,
  "image_size": { "width": 640, "height": 480 },
  "engine": "PP-OCRv5 (NCNN)",
  "backend": "CPU (NCNN NEON FP16)"
}
```

### 2. 提取纯文本

```bash
curl -X POST "http://localhost:8002/api/v1/ocr/text" \
  -F "file=@test.png"
```

---

## 🔧 进阶：启用 RKNN NPU 硬件加速（可选）

泰山派 RK3566 芯片内置 0.8 TOPS NPU。当你获取到 `.rknn` 模型后，无需任何配置，直接放入即可生效：

### 1. 获取 RKNN 模型文件
- **直接下载预转换好的模型（推荐）**：
  从开源社区或瑞芯微官方 Model Zoo 下载针对 RK3566 导出的 `det.rknn` 和 `rec.rknn`。
- **在 PC 端/容器中转换**：
  在 x86 Linux 或 Mac Docker 中执行：
  ```bash
  pip install rknn-toolkit2
  python scripts/convert_to_rknn.py --auto
  ```

### 2. 放入开发板目录
将转换生成的模型放入以下任一目录均可：
- `models/ocr/ppocrv4-rknn/det.rknn` 和 `rec.rknn`
- 或 `models/ocr/ppocrv4/det.rknn` 和 `rec.rknn`

### 3. 重启服务
重新启动服务，系统会自动识别并切换：
```
🚀 [自动检测] 发现 RKNN 模型 (det.rknn, rec.rknn)，已自动启用 NPU 硬件加速推理
```

---

## ⚡ RGA 2D 硬件加速库说明与编译配置

本项目深度适配了瑞芯微 RK3566 芯片的 **RGA 2D 硬件图形加速器**（用于高帧率图像等比缩放与色彩转换），彻底消除大图在 CPU 上逐像素计算的瓶颈：

### 1. 架构与设计
- **DRM DMA-BUF 物理显存分配**：源码位于 [`services/rga_helper.c`](services/rga_helper.c)，基于 Rockchip 官方 `librga` 的 `c_RkRgaGetAllocBuffer` + `wrapbuffer_fd_t` 实现，完美适配 Linux 6.1 内核对硬件连续物理显存的要求；
- **全自动平滑熔断回退**：当尺寸小于 16px、大于 2048px，或缩放倍率超出 1/16~16 倍，或缺少 `/dev/rga` 驱动权限时，系统会自动、静默回退至 CPU OpenCV，确保 100% 稳定运行。

### 2. 编译指南（自动 vs 手动）
- **自动编译（开箱即用）**：当服务启动或首次执行推理时，[`services/rga_utils.py`](services/rga_utils.py) 会自动检测并在缺少动态库时调用本地 `gcc` 即地编译生成 `services/librga_helper.so`；
- **手动编译命令**：
  ```bash
  # 依赖系统已安装 gcc 与 librga 开发库 (泰山派系统通常已自带 /usr/include/rga 与 /usr/lib/librga.so)
  gcc -shared -fPIC -O3 services/rga_helper.c -lrga -o services/librga_helper.so
  ```

### 3. 设备权限配置
Linux 下普通用户直接访问 `/dev/rga` 需配置权限，有以下两种方式：
- **方式一（推荐，随服务自动配置）**：执行 `sudo ./scripts/install_service.sh` 安装服务时，脚本会自动创建 `/etc/udev/rules.d/99-rga.rules` 规则赋权 `MODE="0666"`；
- **方式二（临时开发调试）**：
  ```bash
  sudo chmod 666 /dev/rga
  ```

---

## 🔄 NCNN 模型转换模块 (ONNX ➔ NCNN)

除了开箱即用的预转换模型外，本项目提供了专用的本地模型转换模块与 CLI 工具：

### 1. 命令行一键转换
```bash
# 批量转换指定目录下的全部 ONNX 模型为 NCNN 模型
python scripts/convert_to_ncnn.py -s models/ocr/ppocrv4 -o models/ocr/ppocrv4-ncnn

# 单独转换指定识别模型 (开启 ARM NEON FP16 加速)
python scripts/convert_to_ncnn.py --rec-onnx models/ocr/ppocrv4/rec.onnx

# 转换参数帮助
python scripts/convert_to_ncnn.py --help
```

### 2. Python 代码中模块化调用
可以在任何自定义脚本中直接复用 [`utils.NCNNConverter`](utils/ncnn_converter.py)：
```python
from utils import NCNNConverter

converter = NCNNConverter(fp16=True, opt_level=2)
# 自动使用预设的动态尺寸参数转换文本识别网络
ok, param_file, bin_file = converter.convert_ppocr_rec("models/ocr/ppocrv4/rec.onnx")
```

---

## 🛠️ 开源 Systemd 系统服务部署

项目提供了自动探测路径的部署脚本 [scripts/install_service.sh](scripts/install_service.sh)，彻底杜绝了硬编码用户名和绝对路径的弊端：

```bash
# 1. 一键自动生成并安装系统服务
sudo ./scripts/install_service.sh

# 2. 启动服务与查看状态
sudo systemctl start tspi-ocr
sudo systemctl status tspi-ocr

# 3. 查看实时运行日志
sudo journalctl -u tspi-ocr -f

# 4. 如需卸载系统服务
sudo ./scripts/install_service.sh --uninstall
```

---

## 📁 目录结构

```
ocr-web/
├── config.py                 # 全局配置 (支持 auto/ncnn/rknn/onnx/paddle 引擎配置)
├── main.py                   # 服务统一入口 (API + Gradio 挂载)
├── gradio_app.py             # 独立 Gradio 前端入口
├── webui.py                  # Gradio 左右分栏工作台与前端标注交互
├── models/
│   ├── ocr/ppocrv5-ncnn/     # PP-OCRv5 NCNN 极速模型 (~11MB, det/rec/cls param/bin)
│   ├── ocr/ppocrv4/          # ONNX 模型与字典 (det.onnx, rec.onnx, cls.onnx, dict.txt)
│   └── ocr/ppocrv4-rknn/     # [可选] RKNN 模型存放目录 (det.rknn, rec.rknn)
├── routes/                   # FastAPI 路由 (ocr / stats)
├── services/                 # OCR 引擎抽象 (NCNN / ONNX / RKNN / Paddle) 与 RGA 加速
│   ├── rga_helper.c          # RGA 2D 硬件缩放 C 语言扩展源码 (DRM DMA-BUF 通道)
│   ├── rga_utils.py          # RGA 驱动调用封装与 OpenCV 安全平滑回退
│   └── ocr_service.py        # 多引擎统一推理抽象基类与各引擎实现
├── utils/                    # 通用工具库 (含 ncnn_converter 转换器等)
├── scripts/
│   ├── download_models.py    # Python 多源模型下载与校验工具 (支持 NCNN 与 ONNX)
│   ├── download_models.sh    # Shell 模型下载脚本 (默认国内镜像源)
│   ├── convert_to_ncnn.py    # ONNX 转 NCNN (带 pnnx 自动提取与 FP16 优化) 脚本
│   ├── test_ocr_direct.py    # 板端底层直接推理测试工具
│   ├── convert_to_rknn.py    # [PC端/容器专用] ONNX 转 RKNN 脚本
│   └── install_service.sh    # [开源友好] Systemd 服务自动探测部署脚本
├── tspi-ocr.service          # Systemd 模板文件
└── requirements.txt          # Python 运行时依赖清单
```
