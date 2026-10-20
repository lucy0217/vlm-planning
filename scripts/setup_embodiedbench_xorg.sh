#!/usr/bin/env bash
set -euo pipefail

# 用法：
# sudo bash setup_embodiedbench_xorg.sh 用户名 GPU编号 DISPLAY编号
#
# 示例：
# sudo bash setup_embodiedbench_xorg.sh x_long32 0 99

# 如果要移除，执行：
# sudo systemctl disable --now embodiedbench-xorg-99.service

TARGET_USER="${1:-x_long32}"
GPU_INDEX="${2:-0}"
DISPLAY_ID="${3:-99}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "错误：请使用 sudo/root 运行此脚本。"
    exit 1
fi

for command_name in nvidia-smi Xorg xauth openssl systemctl runuser; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "错误：缺少命令 ${command_name}"
        exit 1
    fi
done

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
    echo "错误：用户 ${TARGET_USER} 不存在。"
    exit 1
fi

TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
TARGET_GROUP="$(id -gn "${TARGET_USER}")"
DISPLAY_NAME=":${DISPLAY_ID}"
SERVICE_NAME="embodiedbench-xorg-${DISPLAY_ID}.service"

CONFIG_FILE="/etc/X11/xorg-embodiedbench-${DISPLAY_ID}.conf"
AUTH_FILE="${TARGET_HOME}/.Xauthority-embodiedbench-${DISPLAY_ID}"
ENV_DIR="${TARGET_HOME}/.config/embodiedbench"
ENV_FILE="${ENV_DIR}/xorg-env.sh"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}"
XORG_BIN="$(command -v Xorg)"

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo "${SERVICE_NAME} 已经在运行，无需重复配置。"
    exit 0
fi

if [[ -e "/tmp/.X11-unix/X${DISPLAY_ID}" ]]; then
    echo "错误：DISPLAY ${DISPLAY_NAME} 已被占用，请换一个编号。"
    exit 1
fi

PCI_RAW="$(
    nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-gpu=pci.bus_id \
        --format=csv,noheader |
    head -n 1 |
    tr -d '[:space:]'
)"

if [[ -z "${PCI_RAW}" ]]; then
    echo "错误：找不到 GPU ${GPU_INDEX}。"
    exit 1
fi

# 例如 00000000:17:00.0 -> PCI:23:0:0
PCI_NO_DOMAIN="${PCI_RAW#*:}"
IFS=':.' read -r BUS_HEX DEVICE_HEX FUNCTION_HEX <<<"${PCI_NO_DOMAIN}"

printf -v XORG_BUS_ID 'PCI:%d:%d:%d' \
    "$((16#${BUS_HEX}))" \
    "$((16#${DEVICE_HEX}))" \
    "$((16#${FUNCTION_HEX}))"

GPU_NAME="$(
    nvidia-smi \
        --id="${GPU_INDEX}" \
        --query-gpu=name \
        --format=csv,noheader |
    head -n 1
)"

echo "配置如下："
echo "  用户：${TARGET_USER}"
echo "  GPU：${GPU_INDEX} (${GPU_NAME})"
echo "  PCI：${PCI_RAW} -> ${XORG_BUS_ID}"
echo "  DISPLAY：${DISPLAY_NAME}"

cat >"${CONFIG_FILE}" <<EOF
Section "ServerFlags"
    Option "AutoAddGPU" "false"
    Option "AutoAddDevices" "false"
EndSection

Section "Device"
    Identifier "EmbodiedBenchGPU"
    Driver "nvidia"
    BusID "${XORG_BUS_ID}"
    Option "AllowEmptyInitialConfiguration" "True"
    Option "UseDisplayDevice" "None"
EndSection

Section "Screen"
    Identifier "EmbodiedBenchScreen"
    Device "EmbodiedBenchGPU"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual 1024 768
    EndSubSection
EndSection

Section "ServerLayout"
    Identifier "EmbodiedBenchLayout"
    Screen 0 "EmbodiedBenchScreen" 0 0
EndSection
EOF

chown root:root "${CONFIG_FILE}"
chmod 0644 "${CONFIG_FILE}"

install -o "${TARGET_USER}" -g "${TARGET_GROUP}" -m 0600 \
    /dev/null "${AUTH_FILE}"

COOKIE="$(openssl rand -hex 16)"

runuser -u "${TARGET_USER}" -- \
    xauth -f "${AUTH_FILE}" \
    add "${DISPLAY_NAME}" MIT-MAGIC-COOKIE-1 "${COOKIE}"

install -d -o "${TARGET_USER}" -g "${TARGET_GROUP}" -m 0700 "${ENV_DIR}"

cat >"${ENV_FILE}" <<EOF
export DISPLAY=${DISPLAY_NAME}
export XAUTHORITY=${AUTH_FILE}
export EB_ALFRED_X_DISPLAY=${DISPLAY_ID}
EOF

chown "${TARGET_USER}:${TARGET_GROUP}" "${ENV_FILE}"
chmod 0600 "${ENV_FILE}"

cat >"${UNIT_FILE}" <<EOF
[Unit]
Description=GPU Xorg ${DISPLAY_NAME} for EmbodiedBench
After=systemd-udev-settle.service nvidia-persistenced.service
Wants=systemd-udev-settle.service

[Service]
Type=simple
ExecStart=${XORG_BIN} ${DISPLAY_NAME} -noreset -nolisten tcp +extension GLX +extension RANDR +extension RENDER -auth ${AUTH_FILE} -config ${CONFIG_FILE} -logfile /var/log/Xorg.embodiedbench.${DISPLAY_ID}.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

chown root:root "${UNIT_FILE}"
chmod 0644 "${UNIT_FILE}"

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

sleep 3

if systemctl is-active --quiet "${SERVICE_NAME}"; then
    echo
    echo "Xorg 已成功启动：${DISPLAY_NAME}"
    echo "服务：${SERVICE_NAME}"
    echo "用户运行评测前执行："
    echo "  source ${ENV_FILE}"
else
    echo "Xorg 启动失败，最近日志如下："
    journalctl -u "${SERVICE_NAME}" -n 100 --no-pager
    exit 1
fi