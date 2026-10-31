#!/usr/bin/env bash
# ==============================================================================
# 泰山派 RK3566 OCR 服务 - Systemd 系统服务一键部署脚本
# 自动探测当前项目路径、运行用户与虚拟环境，生成无硬编码的 systemd 配置文件
# ==============================================================================

set -e

# 确保以 root / sudo 权限运行
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ 权限不足: 请使用 sudo 运行此安装脚本"
    echo "   示例: sudo ./scripts/install_service.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="tspi-ocr"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# 卸载逻辑
if [ "$1" == "--uninstall" ] || [ "$1" == "-u" ]; then
    echo "🛑 正在停止并卸载 ${SERVICE_NAME} 服务..."
    systemctl stop ${SERVICE_NAME} 2>/dev/null || true
    systemctl disable ${SERVICE_NAME} 2>/dev/null || true
    rm -f ${SERVICE_FILE}
    systemctl daemon-reload
    echo "✅ 服务卸载成功！"
    exit 0
fi

# 自动探测运行用户与组 (优先使用执行 sudo 的原始用户)
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "$RUN_USER" 2>/dev/null || echo "$RUN_USER")"

# 探测 Python 执行路径
if [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="$(command -v python3)"
else
    echo "❌ 未检测到 Python 环境，请先创建虚拟环境: python3 -m venv venv"
    exit 1
fi

echo "=================================================="
echo " 📸 泰山派 OCR 服务 Systemd 自动部署"
echo "=================================================="
echo " 项目目录:   $PROJECT_DIR"
echo " 运行用户:   $RUN_USER:$RUN_GROUP"
echo " Python解释器: $PYTHON_EXEC"
echo " 服务文件名: $SERVICE_FILE"
echo "=================================================="

# 生成目标 service 文件
cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=泰山派 RK3566 离线 OCR 文字识别服务
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON_EXEC} main.py
Restart=on-failure
RestartSec=3s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tspi-ocr

# 资源限制优化 (适配板端 2GB 内存)
MemoryMax=800M
CPUQuota=300%

# 环境变量设置
Environment=PYTHONUNBUFFERED=1
Environment=OCR_HOST=0.0.0.0
Environment=OCR_PORT=8002
Environment=OCR_ENGINE=auto

# 安全防护策略
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
EOF

# 赋权并重载 systemd
chmod 644 "$SERVICE_FILE"
systemctl daemon-reload

# 自动配置 /dev/rga 权限规则，让普通用户免 sudo 享受 RGA 2D 硬件加速
if [ -c "/dev/rga" ]; then
    chmod 666 /dev/rga 2>/dev/null || true
    cat <<'RGA_RULE' > /etc/udev/rules.d/99-rga.rules
KERNEL=="rga", MODE="0666"
RGA_RULE
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --name-match=rga 2>/dev/null || true
fi

echo "✅ Systemd 服务配置生成完成！"
echo ""
echo "常用运维命令:"
echo "  启动服务:   sudo systemctl start ${SERVICE_NAME}"
echo "  查看状态:   sudo systemctl status ${SERVICE_NAME}"
echo "  开机自启:   sudo systemctl enable ${SERVICE_NAME}"
echo "  实时日志:   sudo journalctl -u ${SERVICE_NAME} -f"
echo "  卸载服务:   sudo ./scripts/install_service.sh --uninstall"
echo ""
