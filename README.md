# 📸 泰山派 RK3566 离线 OCR 服务

专为 **泰山派 (RK3566, 4核 Cortex-A55 + 2GB 内存)** 打造的高性能离线图片文字识别服务。

---

## 💡 架构设计说明（必读）

本项目内置**智能多引擎自动侦测机制**，默认开箱即用：

```
                             ┌──> [检测到 RKNN 模型] ──> NPU 硬件加速推理 (0.8 TOPS RKNNLite, ~80ms)
                             │
                             ├──> [检测到 NCNN 模型] ──> PP-OCRv5 NCNN 极速引擎 (NEON FP16, ~150ms) ⭐推荐
 上传图片 ──> 启动自动检测 ──┤
                             ├──> [选择 Indra 引擎] ──> Indra OCR 引擎 (1M极轻量, 80+语言+CJK, ~1.0s)
                             │
                             └──> [回退 ONNX 引擎] ──> ONNX 通用引擎 (4核 CPU 并行 + RGA 硬件缩放, ~1.5s)
```

1. **智能多引擎自动分级与手动切换**：
   - 默认配置为 `OCR_ENGINE=auto`；
   - 服务启动时自动扫描模型目录并依优先级加载：
     1. **优先检测 RKNN 模型**（`models/ocr/ppocrv4-rknn/`）：若存在 `det.rknn` 与 `rec.rknn`，优先启用板载 0.8 TOPS NPU 硬件加速；
     2. **其次检测 NCNN 模型**（`models/ocr/ppocrv5-ncnn/`）：若存在 PP-OCRv5 NCNN 模型，自动启用 ARM NEON FP16 汇编极速引擎，模型仅 11MB，推理耗时大幅缩短至 **150ms 左右**；
     3. **支持 Indra OCR 超轻量多语言模型**（`models/ocr/indra/`）：检测模型仅 **389 KB**，支持 80+ 种语言与 11,749 个 CJK 中文字符识别，版面排版解析精准；
     4. **兜底回退 ONNX 引擎**（`models/ocr/ppocrv4/`）：若以上均未找到，无缝平滑回退至通用 ONNX 引擎。
2. **PC 端转换 vs 板端推理分工**：
   - 瑞芯微官方模型转换工具（`rknn-toolkit2`）依赖 x86 交叉编译库，**仅可在 x86_64 Linux PC / 容器中运行**；
   - 泰山派板端仅运行推理库（`rknn-toolkit-lite2`、`ncnn` 与 `onnxruntime`），负责加载模型进行高帧率低延迟推理。

---

## 📋 核心特性

- **多引擎智能融合**：优先 NPU 硬件加速（RKNN），秒级平滑分级（NCNN NEON FP16 -> Indra OCR -> ONNX）。
- **极速响应**：PP-OCRv5 NCNN 引擎整图识别约 **120~180ms** + RGA 2D 硬件缩放辅助。
- **超小内存占用**：PP-OCRv5 NCNN 模型总体量仅 **11MB**；Indra 检测网络仅 **389KB**，极大节约板载内存。
- **完全离线运行**：本地私有化部署，不依赖外部网络连接与第三方 API。
- **零磁盘 IO**：图片全流程内存解码处理，不产生临时文件、不损耗板载 eMMC/TF 闪存寿命。
- **全方位置信度呈现**：
  - **顶部实时指标条**：展示整图 `🎯 平均置信度: XX%`、推理耗时、行数与分辨率；
  - **结果双 Tab 切换**：支持「📝 识别纯文本」一键复制，与「📊 逐行文字与置信度明细」表格（直观呈现每行文字、置信度百分比与质量评级 🟢/🟡/🟠）；
  - **客户端标注框高亮**：悬浮标签置信度前置（如 `[98%] #1 识别文字`），支持鼠标交互查看。
- **单端口全栈部署**：FastAPI RESTful 后端与 Gradio Web UI 共用端口 `8002`。
- **板端多引擎对比评测工具**：内置 `scripts/compare_engines.py`，支持一键横向评测对比不同引擎在实物图片上的准确率与耗时。

---

## 📊 引擎实测性能对比 (RK3566 实测)

以测试图片（473×268 截屏图片）为例，在泰山派 RK3566（4核 A55 @ 1.8GHz）上的实测数据：

| 对比维度 | **PP-OCRv5 NCNN** | **Indra OCR** | **PP-OCRv4 ONNX** |
| :--- | :--- | :--- | :--- |
| **模型设计机构** | 百度飞桨官方团队 | Datalab (1M 参数极轻量架构) | 百度飞桨官方团队 |
| **总模型大小** | **~11 MB** (det: 2.5MB, rec: 8.5MB) | **~10 MB** (det: **389 KB**, rec: 9.7MB) | ~90 MB |
| **检测模型体量** | 2.5 MB | **仅 389 KB** (超轻量) | ~4.5 MB |
| **推理后端** | NCNN (ARM NEON FP16 汇编) | ONNX Runtime (4核 CPU 并行) | ONNX Runtime (4核 CPU 并行) |
| **整图识别耗时** | 约 **150 ~ 1700 ms** (取决于行数) | 约 **1000 ~ 1100 ms** | 约 1500 ~ 2200 ms |
| **字符集支持** | 6600+ 常用中文 + 英文数字 | 80+ 语言 + 11,749 CJK 中日韩字符 | 6600+ 常用中文 + 英文数字 |
| **推荐适用场景** | 中文自然场景、广告票据、大字识别 | 多语言混合、PDF复杂版面、极小内存部署 | 通用平台跨平台保底与对照 |

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

本项目内置自动下载工具，支持一键拉取国内镜像源模型：

```bash
# 1. 默认模式：自动检查并下载推荐的 PP-OCRv5 NCNN 与 Indra 模型
python scripts/download_models.py

# 2. 仅下载/更新指定模型：
python scripts/download_models.py --ncnn       # 仅下载 PP-OCRv5 NCNN 极速模型 (~11MB)
python scripts/download_models.py --indra      # 仅下载 Indra OCR 1M 超轻量模型 (~10MB)
python scripts/download_models.py --onnx       # 仅下载 PP-OCRv4 原版 ONNX 模型 (~90MB)
python scripts/download_models.py --all        # 下载全部模型包 (NCNN + ONNX + Indra)
python scripts/download_models.py --force      # 强制重新拉取覆盖
```

#### 📦 开源模型仓库与资源索引
1. **[qefro/pp-OCRv5-6-ncnn (HuggingFace)](https://huggingface.co/qefro/pp-OCRv5-6-ncnn)**：
   - PP-OCRv5 移动端精简模型，经过 NEON FP16 深度优化；
   - 包含检测 (`det.bin` 2.3MB)、识别 (`rec.bin` 7.9MB) 与方向分类 (`cls.bin` 475KB)。
2. **[datalab-to/indra-ocr (HuggingFace)](https://huggingface.co/datalab-to/indra-ocr)**：
   - 官方 1M 参数轻量多语言 OCR 模型；
   - 检测网络仅 389KB，识别网络原生覆盖 80+ 全球语种及 11,749 个 CJK 汉字字符。
3. **[monkt/paddleocr-onnx (HuggingFace)](https://huggingface.co/monkt/paddleocr-onnx)**：
   - 原版 PP-OCRv4 ONNX 格式模型（~90MB），适合作为跨平台保底。

### 3. 一键横向对比评测（可选）

在板端运行以下命令，即可针对任何图片横向输出各引擎的识别耗时、文本行数与置信度对比：

```bash
# 评测当前目录的 test_chinese.png
python scripts/compare_engines.py

# 指定自定义测试图片并指定测试轮数
python scripts/compare_engines.py --image path/to/image.png --rounds 3
```

### 4. 启动服务

```bash
# 推荐：单端口同时启动 API + Web UI (端口 8002)
python main.py

# 其它启动模式选项:
python main.py --api-only           # 仅启动 RESTful API
python main.py --port 8002          # 自定义端口
python gradio_app.py --port 7861     # 独立启动 Gradio 前端
```

### 5. 访问服务

- **Web UI 工作台**: `http://<开发板IP>:8002/gradio/`
- **Swagger API 文档**: `http://<开发板IP>:8002/docs`
- **健康监控**: `http://<开发板IP>:8002/health`

---

## 📡 API 调用示例

### 1. 图片 OCR 识别（支持指定引擎与 RGA 开关）

```bash
# 使用推荐的 NCNN 极速引擎
curl -X POST "http://localhost:8002/api/v1/ocr/recognize?engine=ncnn&use_rga=true" \
  -F "file=@test.png" \
  -F "language=ch"

# 使用 Indra 超轻量多语言引擎
curl -X POST "http://localhost:8002/api/v1/ocr/recognize?engine=indra&use_rga=true" \
  -F "file=@test.png"
```

**返回示例**：
```json
{
  "success": true,
  "text": "恭喜您获得618狂欢节1元基金\n知道了",
  "results": [
    {
      "bbox": [128.6, 54.8, 343.4, 54.8, 343.4, 71.5, 128.6, 71.5],
      "text": "恭喜您获得618狂欢节1元基金",
      "confidence": 0.984
    },
    {
      "bbox": [201.2, 218.4, 271.8, 218.4, 271.8, 241.6, 201.2, 241.6],
      "text": "知道了",
      "confidence": 1.000
    }
  ],
  "processing_time": 0.148,
  "image_size": { "width": 473, "height": 268 },
  "engine": "Indra-OCR",
  "backend": "ONNX Runtime (CPU 4核)"
}
```

### 2. 纯文本快捷提取

```bash
curl -X POST "http://localhost:8002/api/v1/ocr/text" \
  -F "file=@test.png"
```

---

## ⚡ RGA 2D 硬件加速库说明与编译配置

本项目深度适配了瑞芯微 RK3566 芯片的 **RGA 2D 硬件图形加速器**（用于高帧率图像等比缩放与色彩转换），彻底消除大图在 CPU 上逐像素计算的瓶颈：

### 1. 架构与设计
- **DRM DMA-BUF 物理显存分配**：源码位于 [`services/rga_helper.c`](services/rga_helper.c)，基于 Rockchip 官方 `librga` 的 `c_RkRgaGetAllocBuffer` + `wrapbuffer_fd_t` 实现，完美适配 Linux 6.1 内核对硬件连续物理显存的要求；
- **全自动平滑熔断回退**：当尺寸小于 16px、大于硬件上限或无显存通道时，无感知降级到 OpenCV 多线程处理；
- **动态库编译安装**：执行 `./scripts/install_service.sh` 会自动检测并编译 `services/rga_helper.so`。

---

## 🛠️ ONNX ➔ NCNN 模型转换工具

项目内置了完整的 ONNX ➔ NCNN 转换流水线：

```bash
# 批量转换指定目录下的全部 ONNX 模型为 NCNN 模型
python scripts/convert_to_ncnn.py -s models/ocr/ppocrv4 -o models/ocr/ppocrv4-ncnn

# 单独转换指定识别模型 (开启 ARM NEON FP16 加速)
python scripts/convert_to_ncnn.py --rec-onnx models/ocr/ppocrv4/rec.onnx

# 查看参数帮助
python scripts/convert_to_ncnn.py --help
```

- 转换核心模块：[`utils/ncnn_converter.py`](utils/ncnn_converter.py)
- Indra PyTorch 结构代码参考：[`utils/indra_models/`](utils/indra_models/)

---

## 🛠️ 开源 Systemd 系统服务部署

项目提供了自动探测路径的部署脚本 [scripts/install_service.sh](scripts/install_service.sh)：

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
├── config.py                 # 全局配置 (支持 auto/ncnn/rknn/indra/onnx/paddle 引擎配置)
├── main.py                   # 服务统一入口 (API + Gradio 挂载)
├── gradio_app.py             # 独立 Gradio 前端入口
├── webui.py                  # Gradio 现代化工作台 (双 Tab 置信度表格 + 悬浮标注)
├── models/                   # 模型存储目录 (默认由 download_models.py 自动下载拉取)
│   ├── ocr/ppocrv5-ncnn/     # PP-OCRv5 NCNN 极速模型 (~11MB, det/rec/cls param/bin)
│   ├── ocr/indra/            # Indra OCR 1M 超轻量模型 (det.onnx 389KB, rec_cjk.onnx, meta.json)
│   ├── ocr/ppocrv4/          # PP-OCRv4 ONNX 模型与字典
│   └── ocr/ppocrv4-rknn/     # [可选] RKNN NPU 模型存放目录 (det.rknn, rec.rknn)
├── routes/                   # FastAPI 路由 (ocr / stats)
├── services/                 # OCR 引擎实现 (NCNN / Indra / ONNX / RKNN / Paddle) 与 RGA 加速
│   ├── rga_helper.c          # RGA 2D 硬件缩放 C 语言扩展源码 (DRM DMA-BUF 通道)
│   ├── rga_utils.py          # RGA 驱动调用封装与安全平滑回退
│   └── ocr_service.py        # 多引擎统一抽象与算法调度实现
├── utils/                    # 通用工具库
│   ├── ncnn_converter.py     # ONNX -> NCNN 模型转换专用模块
│   └── indra_models/         # Indra 网络结构 PyTorch 定义代码 (det_model, rec_model)
├── scripts/
│   ├── download_models.py    # Python 多源模型下载与校验工具 (支持 NCNN/Indra/ONNX)
│   ├── download_models.sh    # Shell 模型下载脚本
│   ├── compare_engines.py    # 多 OCR 引擎横向性能与准确率对比评测工具
│   ├── convert_to_ncnn.py    # ONNX 转 NCNN CLI 命令行工具
│   ├── test_ocr_direct.py    # 板端底层直接推理测试脚本
│   ├── convert_to_rknn.py    # [PC端/容器专用] ONNX 转 RKNN 脚本
│   └── install_service.sh    # Systemd 服务自动探测部署脚本
├── tspi-ocr.service          # Systemd 服务模板文件
└── requirements.txt          # Python 运行时依赖清单
```
