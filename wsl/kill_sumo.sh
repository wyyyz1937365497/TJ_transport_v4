#!/bin/bash
# SUMO进程清理脚本
# 用于清理所有SUMO相关进程和端口

echo "======================================"
echo "SUMO进程清理脚本"
echo "======================================"

# 杀死所有SUMO进程
echo "正在杀死SUMO进程..."
killall -9 sumo 2>/dev/null
killall -9 sumo-gui 2>/dev/null
killall -9 sumo.exe 2>/dev/null

# 等待进程完全终止
sleep 2

# 检查是否还有残留进程
REMAINING=$(ps aux | grep -E "sumo" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "警告: 仍有 $REMAINING 个SUMO进程运行"
    ps aux | grep -E "sumo" | grep -v grep
else
    echo "所有SUMO进程已清理"
fi

# 检查端口占用
echo ""
echo "检查SUMO常用端口 (8813-8850)..."

for port in {8813..8850}; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "  端口 $port 被占用:"
        lsof -Pi :$port -sTCP:LISTEN
    fi
done

echo ""
echo "======================================"
echo "清理完成"
echo "======================================"
