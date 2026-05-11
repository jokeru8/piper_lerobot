#!/bin/bash
# 激活两个已经通过 udev 命名好的 CAN 接口
BITRATE=1000000

for iface in can_master can_follower; do
    if ! ip link show "$iface" &>/dev/null; then
        echo "Error: $iface not found. Check USB connection and udev rules."
        exit 1
    fi
    sudo ip link set "$iface" down 2>/dev/null
    sudo ip link set "$iface" type can bitrate $BITRATE
    sudo ip link set "$iface" up
    echo "$iface activated at ${BITRATE} bps"
done
