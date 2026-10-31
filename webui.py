"""
泰山派 RK3566 OCR 服务 - Gradio Web UI 模块
现代双栏工作台布局 · 客户端原生标注渲染 · 毫秒级离线推理
"""

import os
import io
import json
import sys
import urllib.request
from pathlib import Path

# 禁用遥测（避免无外网环境网络阻塞）
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr
import numpy as np
from PIL import Image

from config import settings

# ============ API 基础地址 ============
DEFAULT_API_BASE = f"http://localhost:{settings.PORT}"
API_BASE = os.environ.get("OCR_API_BASE", DEFAULT_API_BASE)


# ============ HTTP 请求辅助函数 ============
def api_request(endpoint, method="GET", data=None, files=None, api_base=None):
    """发送 HTTP 请求到后端 API 并返回解析后的 JSON"""
    import uuid as _uuid

    base = (api_base or API_BASE).rstrip("/")
    url = f"{base}{endpoint}"
    try:
        if method == "GET":
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
        else:
            if files:
                boundary = _uuid.uuid4().hex.encode()
                sep = b"\r\n"
                body_parts = []
                for key, (filename, file_data, mime) in files.items():
                    header = (
                        b"--" + boundary + sep +
                        (f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n').encode() +
                        (f"Content-Type: {mime}\r\n").encode() +
                        sep
                    )
                    body_parts.append(header)
                    body_parts.append(file_data if isinstance(file_data, bytes) else file_data.encode())
                    body_parts.append(sep)
                body_parts.append(b"--" + boundary + b"--" + sep)
                body = b"".join(body_parts)
                headers = {"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"}
            else:
                body = json.dumps(data).encode("utf-8") if data else None
                headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(url, data=body, headers=headers, method=method)

        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read()
            try:
                return json.loads(raw)
            except Exception:
                return {"raw": raw[:500].decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        try:
            err_json = json.loads(body)
            return {"error": err_json.get("detail", body.decode("utf-8", errors="replace"))}
        except Exception:
            return {"error": body.decode("utf-8", errors="replace")[:500] or f"HTTP 错误 {e.code}"}
    except Exception as e:
        return {"error": str(e)}


# ============ 客户端渲染格式转换 ============
def format_annotations(image, results):
    """
    将 OCR 标注转换为前端 AnnotatedImage 格式 (零板端绘图开销)
    格式: (base_image, [((x_min, y_min, x_max, y_max), label), ...])
    由浏览器前端利用 DOM/Canvas 高亮渲染，板端零 CPU 画图损耗
    """
    if image is None:
        return None
    if not results:
        return (image, [])

    annotations = []
    for i, r in enumerate(results):
        bbox = r.get("bbox", [])
        text = r.get("text", "").strip()
        conf = r.get("confidence", 0.0)
        # 精简标签展示，避免长串文字生成庞大标签块遮挡原图
        short_text = (text[:10] + "...") if len(text) > 10 else text
        label = f"#{i+1} {short_text} ({conf:.0%})" if short_text else f"#{i+1} ({conf:.0%})"

        if len(bbox) >= 8:
            x_min = int(min(bbox[0], bbox[2], bbox[4], bbox[6]))
            y_min = int(min(bbox[1], bbox[3], bbox[5], bbox[7]))
            x_max = int(max(bbox[0], bbox[2], bbox[4], bbox[6]))
            y_max = int(max(bbox[1], bbox[3], bbox[5], bbox[7]))
            annotations.append(((x_min, y_min, x_max, y_max), label))
        elif len(bbox) == 4:
            annotations.append(((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), label))

    return (image, annotations)


# ============ OCR 识别主流程 ============
def ocr_recognize_workflow(image, url, input_mode, language, det, rec, cls, engine_choice, use_rga, api_base):
    """
    统一 OCR 识别流程
    返回: (annotated_image, text_result, metrics_markdown, json_code)
    """
    target_img = None
    img_bytes = None

    if input_mode == "图片 URL":
        if not url or not url.strip():
            return None, "", "⚠️ 请输入有效的图片 URL 地址", "{}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                img_bytes = resp.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            target_img = np.array(pil_img)
        except Exception as e:
            return None, "", f"❌ 图片下载失败: {e}", "{}"
    else:
        if image is None:
            return None, "", "⚠️ 请先上传图片或使用剪贴板粘贴截图 (Ctrl+V)", "{}"

        # 兼容多种图片输入格式 (numpy / dict / str path / PIL)
        try:
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
                target_img = np.array(pil_img)
            elif isinstance(image, dict) and "path" in image:
                pil_img = Image.open(image["path"]).convert("RGB")
                target_img = np.array(pil_img)
            elif isinstance(image, np.ndarray):
                target_img = image
                pil_img = Image.fromarray(image.astype("uint8")).convert("RGB")
            elif hasattr(image, "convert"):
                pil_img = image.convert("RGB")
                target_img = np.array(pil_img)
            else:
                return None, "", "⚠️ 无法解析上传的图片格式", "{}"

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        except Exception as e:
            return None, "", f"❌ 图片解析编码失败: {e}", "{}"

    # 映射语言与引擎参数
    lang_map = {"中文": "ch", "英文": "en", "日文": "japan", "韩文": "korean"}
    lang_param = lang_map.get(language, "ch")

    engine_map = {
        "智能自动 (优先 RKNN/NCNN)": "auto",
        "智能自动 (优先 RKNN)": "auto",
        "PP-OCRv5 (NCNN FP16 极速)": "ncnn",
        "PP-OCRv4 (ONNX CPU 4核)": "onnx",
        "PP-OCRv4 (RKNN NPU加速)": "rknn",
    }
    engine_param = engine_map.get(engine_choice, "auto")

    # 调用后端接口
    query_params = (
        f"?language={lang_param}&det={str(det).lower()}&rec={str(rec).lower()}&cls={str(cls).lower()}"
        f"&engine={engine_param}&use_rga={str(use_rga).lower()}"
    )
    result = api_request(
        f"/api/v1/ocr/recognize{query_params}",
        method="POST",
        files={"file": ("image.png", img_bytes, "image/png")},
        api_base=api_base,
    )

    if "error" in result:
        err_msg = result.get("error", "未知错误")
        return (target_img, []), f"识别失败: {err_msg}", f"❌ 服务端报错: {err_msg}", "{}"

    # 组装返回数据
    results_list = result.get("results", [])
    annotated = format_annotations(target_img, results_list)
    full_text = result.get("text", "")
    duration_ms = round(result.get("processing_time", 0) * 1000, 1)
    line_count = len(results_list)

    img_size = result.get("image_size", {})
    w = img_size.get("width", target_img.shape[1] if hasattr(target_img, "shape") else "-")
    h = img_size.get("height", target_img.shape[0] if hasattr(target_img, "shape") else "-")

    engine_badge = result.get("engine", engine_choice)
    backend_badge = result.get("backend", "CPU")

    metrics_html = (
        f'<div class="metric-container">'
        f'<span class="metric-item">⏱️ 推理耗时: <b>{duration_ms} ms</b></span>'
        f'<span class="metric-item">📝 文本行数: <b>{line_count} 行</b></span>'
        f'<span class="metric-item">📐 图像分辨率: <b>{w}×{h} px</b></span>'
        f'<span class="metric-badge">🟢 {engine_badge}</span>'
        f'</div>'
    )

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    return annotated, full_text, metrics_html, json_str


# ============ 统计信息格式化 ============
def fetch_system_stats(api_base=None):
    """获取并美化展示统计指标"""
    result = api_request("/api/v1/stats/", api_base=api_base)
    if "error" in result:
        return f"❌ 获取失败: {result.get('error', '未知错误')}", "{}"

    uptime = result.get("uptime_human", "-")
    total_calls = result.get("total_calls", 0)
    total_errors = result.get("total_errors", 0)
    err_rate = result.get("overall_error_rate", 0.0)
    success_rate = round(100.0 - err_rate, 1)

    summary_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0;">
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 13px; color: #64748b;">累计调用量</div>
            <div style="font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 4px;">{total_calls} <span style="font-size: 14px; font-weight: 400;">次</span></div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 13px; color: #64748b;">识别成功率</div>
            <div style="font-size: 26px; font-weight: 700; color: #16a34a; margin-top: 4px;">{success_rate}%</div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 13px; color: #64748b;">异常调用次数</div>
            <div style="font-size: 26px; font-weight: 700; color: #dc2626; margin-top: 4px;">{total_errors} <span style="font-size: 14px; font-weight: 400;">次</span></div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center;">
            <div style="font-size: 13px; color: #64748b;">系统持续运行</div>
            <div style="font-size: 20px; font-weight: 700; color: #0284c7; margin-top: 8px;">{uptime}</div>
        </div>
    </div>
    """

    endpoints = result.get("endpoints", {})
    endpoint_lines = []
    if endpoints:
        endpoint_lines.append("### 📌 各端点耗时明细")
        for ep, st in endpoints.items():
            endpoint_lines.append(
                f"- **`{ep}`**：调用 **{st['total_calls']}** 次 | 平均耗时 **{st['avg_duration_ms']}ms** | 最近耗时 **{st['last_duration_ms']}ms**"
            )
    else:
        endpoint_lines.append("*暂无独立端点明细*")

    details_md = summary_html + "\n\n" + "\n".join(endpoint_lines)
    return details_md, json.dumps(result, ensure_ascii=False, indent=2)


# ============ Gradio 界面构建 ============
def build_interface():
    """构建全新现代化工作台布局界面"""

    custom_css = """
    /* 容器全局优化 */
    .gradio-container {
        max-width: 1360px !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    /* 顶栏卡片 */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .header-title {
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-sub {
        font-size: 13px;
        color: #94a3b8;
        margin: 0;
    }
    /* 运行指标条 */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 13px;
        color: #334155;
    }
    .metric-item b {
        color: #0f172a;
    }
    .metric-badge {
        margin-left: auto;
        background: #dcfce7;
        color: #15803d;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    /* 代码卡片 */
    .api-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    /* 解决标注蒙版与标签遮挡图片问题 */
    .image-container .mask {
        opacity: 0.22 !important;
        mix-blend-mode: multiply;
        transition: opacity 0.15s ease-in-out;
    }
    .image-container:hover .mask {
        opacity: 0.35 !important;
    }
    .image-container .mask.active {
        opacity: 0.55 !important;
    }
    /* 图例限制高度并支持内部滚动，禁止遮挡图片 */
    .legend {
        max-height: 75px !important;
        overflow-y: auto !important;
        padding: 4px 6px !important;
        gap: 4px !important;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin-top: 6px;
    }
    .legend-item {
        font-size: 11px !important;
        padding: 2px 6px !important;
        line-height: 1.2 !important;
        max-width: 180px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    """

    with gr.Blocks(title="泰山派 RK3566 离线 OCR", css=custom_css, theme=gr.themes.Soft()) as demo:
        # ---- 顶部导航与状态条 ----
        gr.HTML(
            f"""
            <div class="header-card">
                <div class="header-title">
                    <span>📸 泰山派 RK3566 离线 OCR 文字识别工作台</span>
                </div>
                <div class="header-sub">
                    PP-OCRv4 神经网络 · 4核 CPU 并行加速 · 纯前端交互式标注渲染 · 全流程零磁盘落盘
                </div>
            </div>
            """
        )

        with gr.Accordion("⚙️ 服务端连接设置 (默认直连本机，远程访问可在此修改)", open=False):
            api_base_input = gr.Textbox(
                value=f"http://localhost:{settings.PORT}",
                label="后端 API 地址",
                placeholder="http://192.168.x.x:8002 或 http://localhost:8002",
                info="当网页运行在远端电脑但调用板端 API 时，填入板子 IP:端口",
                lines=1,
            )

        # ============ 主工作台选项卡 ============
        with gr.Tabs():
            # ---- Tab 1: OCR 工作台 (核心主界面，一站式输入与展示) ----
            with gr.TabItem("🚀 OCR 识别工作台"):
                with gr.Row(equal_height=False):
                    # ---- 左侧：输入与控制面板 ----
                    with gr.Column(scale=5):
                        input_mode = gr.Radio(
                            choices=["上传图片 / 剪贴板粘贴", "图片 URL"],
                            value="上传图片 / 剪贴板粘贴",
                            label="输入来源",
                            interactive=True,
                        )

                        image_input = gr.Image(
                            type="numpy",
                            label="图像输入 (支持点击上传、拖拽或 Ctrl+V 粘贴截图)",
                            sources=["upload", "clipboard"],
                            image_mode="RGB",
                            visible=True,
                            height=360,
                        )

                        url_input = gr.Textbox(
                            label="图片网络 URL",
                            placeholder="https://example.com/sample.png",
                            visible=False,
                            lines=2,
                        )

                        with gr.Accordion("⚙️ 模型引擎与高级参数设置", open=False):
                            with gr.Row():
                                engine_choice = gr.Dropdown(
                                    choices=["智能自动 (优先 RKNN/NCNN)", "PP-OCRv5 (NCNN FP16 极速)", "PP-OCRv4 (ONNX CPU 4核)", "PP-OCRv4 (RKNN NPU加速)"],
                                    value="智能自动 (优先 RKNN/NCNN)",
                                    label="🧠 推理模型引擎",
                                    interactive=True,
                                )
                                language = gr.Dropdown(
                                    choices=["中文", "英文", "日文", "韩文"],
                                    value="中文",
                                    label="识别语言",
                                    interactive=True,
                                )
                            with gr.Row():
                                rga_checkbox = gr.Checkbox(value=True, label="⚡ 启用 RGA 硬件加速缩放 (不可用或失败时自动安全回退 OpenCV)")
                                det_checkbox = gr.Checkbox(value=True, label="文本区域检测 (det)")
                                rec_checkbox = gr.Checkbox(value=True, label="文本内容识别 (rec)")
                                cls_checkbox = gr.Checkbox(value=False, label="文字方向校正 (cls)")

                        with gr.Row():
                            recognize_btn = gr.Button("🚀 开始文字识别", variant="primary", scale=2, size="lg")
                            clear_btn = gr.Button("🗑️ 清空", variant="secondary", scale=1, size="lg")

                    # ---- 右侧：结果展示面板 ----
                    with gr.Column(scale=6):
                        # 运行指标状态条
                        metrics_display = gr.HTML(
                            """
                            <div class="metric-container">
                                <span>等待上传图片并点击「开始文字识别」...</span>
                            </div>
                            """
                        )

                        # 原生客户端标注渲染（纯净框选高亮，原图不被遮挡）
                        annotated_preview = gr.AnnotatedImage(
                            label="标注可视化 (半透明框选，原图文字清晰可见)",
                            height=420,
                            show_legend=False,
                        )

                        # 纯文本结果（直接复制）
                        text_output = gr.Textbox(
                            label="识别文字结果 (支持直接编辑与一键复制)",
                            lines=7,
                            placeholder="识别出的完整文本内容将呈现在这里...",
                            show_copy_button=True,
                        )

                        with gr.Accordion("📋 结构化数据 (JSON)", open=False):
                            json_output = gr.Code(label="包含各检测框 8 点坐标的原始数据", language="json", lines=8)

                # 交互逻辑绑定
                def switch_input_view(mode):
                    is_upload = mode == "上传图片 / 剪贴板粘贴"
                    return gr.update(visible=is_upload), gr.update(visible=not is_upload)

                input_mode.change(switch_input_view, inputs=[input_mode], outputs=[image_input, url_input])

                recognize_btn.click(
                    fn=ocr_recognize_workflow,
                    inputs=[
                        image_input,
                        url_input,
                        input_mode,
                        language,
                        det_checkbox,
                        rec_checkbox,
                        cls_checkbox,
                        engine_choice,
                        rga_checkbox,
                        api_base_input,
                    ],
                    outputs=[annotated_preview, text_output, metrics_display, json_output],
                )

                clear_btn.click(
                    fn=lambda: (
                        None,
                        "",
                        None,
                        "",
                        '<div class="metric-container"><span>已清空内容</span></div>',
                        "{}",
                    ),
                    outputs=[image_input, url_input, annotated_preview, text_output, metrics_display, json_output],
                )

            # ---- Tab 2: API 开发者对接 ----
            with gr.TabItem("🔌 API 开发者中心"):
                gr.Markdown(
                    f"""
                    ### 📡 RESTful API 快速集成指南
                    后端接口与 Web UI 共用端口，返回纯 JSON YOLO 标注坐标与文本。

                    **在线交互式文档 (Swagger UI)**: [http://localhost:{settings.PORT}/docs](http://localhost:{settings.PORT}/docs)
                    """
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            f"""
                            #### 1. 通用文字识别 (POST `/api/v1/ocr/recognize`)
                            上传图片文件，返回包含坐标 `bbox`、`text` 及 `confidence` 的结构化数据：

                            ```bash
                            curl -X POST "{API_BASE}/api/v1/ocr/recognize" \\
                              -F "file=@screenshot.png" \\
                              -F "language=ch"
                            ```
                            """
                        )
                    with gr.Column():
                        gr.Markdown(
                            f"""
                            #### 2. 纯文本提取 (POST `/api/v1/ocr/text`)
                            仅需提取纯字符串内容，省去解析坐标步骤：

                            ```bash
                            curl -X POST "{API_BASE}/api/v1/ocr/text" \\
                              -F "file=@document.jpg"
                            ```
                            """
                        )

            # ---- Tab 3: 系统监控与调用统计 ----
            with gr.TabItem("📊 系统监控与统计"):
                with gr.Row():
                    gr.Markdown("### 实时服务调用与监控看板")
                    refresh_stats_btn = gr.Button("🔄 刷新最新统计", variant="secondary", size="sm")

                stats_display = gr.HTML(
                    """
                    <div style="padding: 20px; text-align: center; color: #64748b;">
                        点击上方「刷新最新统计」查看调用指标看板
                    </div>
                    """
                )

                with gr.Accordion("原始统计 JSON", open=False):
                    stats_raw_json = gr.Code(label="JSON 数据", language="json", lines=10)

                refresh_stats_btn.click(
                    fn=fetch_system_stats,
                    inputs=[api_base_input],
                    outputs=[stats_display, stats_raw_json],
                )

            # ---- Tab 4: 硬件与技术架构说明 ----
            with gr.TabItem("ℹ️ 架构与硬件"):
                gr.Markdown(
                    f"""
                    ## 泰山派 RK3566 离线 OCR 架构设计

                    ### 硬件规格与运行环境
                    - **主控芯片**: Rockchip RK3566 (Quad-core Cortex-A55 @ 1.8GHz)
                    - **板载内存**: 2GB LPDDR4
                    - **硬件加速**: RGA (2D 图像硬件引擎) + Mali-G52 GPU + 0.8 TOPS NPU

                    ### 架构设计优势
                    1. **纯前端标注渲染 (Zero-Server-Draw)**:
                       - 传统方式：后端通过 PIL 绘制线框、加载字体、压缩大图，极度消耗板端 CPU；
                       - 本系统：后端仅返回轻量 JSON 坐标，由浏览器客户端 GPU / Canvas 本地渲染高亮框，板端零绘图开销。
                    2. **多线程 CPU 并行**:
                       - 采用轻量 ONNX Runtime，充分利用 RK3566 的 4 颗 A55 核心进行算子级多线程并行，完全脱离庞大框架依赖。
                    3. **全流程内存处理 (Zero-Disk-IO)**:
                       - 客户端上传图片字节流通过内存直接解码并传递给推理引擎，不产生任何临时文件，大幅保护闪存寿命并确保数据隐私。
                    """
                )

        gr.HTML(
            """
            <div style="text-align: center; color: #94a3b8; font-size: 12px; margin-top: 24px; padding-bottom: 12px;">
                © 2026 泰山派智能 OCR · 完全离线部署 · 保护隐私安全
            </div>
            """
        )

    return demo


# ============ 独立启动入口 ============
def launch(port=None):
    """独立启动 Gradio 前端进程"""
    ui_port = port or settings.WEBUI_PORT
    demo = build_interface()
    demo.launch(
        server_name=settings.HOST,
        server_port=ui_port,
        share=False,
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    launch()
