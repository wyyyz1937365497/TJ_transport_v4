#!/bin/bash
# 监控数据收集进度的脚本

echo "========================================"
echo "数据收集进度监控"
echo "========================================"
echo ""

# 检查进程是否还在运行
if ps -p 2899 > /dev/null 2>&1; then
    echo "✓ 数据收集进程运行中 (PID: 2899)"
    echo ""
else
    echo "✗ 数据收集进程已结束"
    echo ""
    echo "最后20行日志："
    tail -20 data_collection.log
    exit 1
fi

# 显示实时日志
echo "最近日志："
echo "----------------------------------------"
tail -30 data_collection.log

echo ""
echo "========================================"
echo "输入以下命令查看实时日志："
echo "  tail -f data_collection.log"
echo ""
echo "输入以下命令停止收集："
echo "  kill 2899"
echo "========================================"
