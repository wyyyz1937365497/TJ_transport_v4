#!/usr/bin/env python3
"""
训练性能监控脚本

实时监控GPU和CPU占用情况，帮助识别性能瓶颈
"""

import time
import psutil
import subprocess
import json
from datetime import datetime

def get_gpu_stats():
    """获取GPU统计信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'id': int(parts[0]),
                    'gpu_util': float(parts[1]),
                    'mem_util': float(parts[2]),
                    'mem_used': float(parts[3]),
                    'mem_total': float(parts[4]),
                    'power': float(parts[5])
                })
        return gpus
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []

def get_cpu_stats():
    """获取CPU统计信息"""
    try:
        # 找到Python训练进程
        python_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] == 'python' and proc.info['cpu_percent'] > 5:
                    python_procs.append({
                        'pid': proc.info['pid'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'python_procs': python_procs
        }
    except Exception as e:
        print(f"获取CPU信息失败: {e}")
        return {}

def print_stats(gpu_stats, cpu_stats):
    """打印统计信息"""
    print("\n" + "="*80)
    print(f"训练性能监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # GPU信息
    print("\n【GPU状态】")
    for gpu in gpu_stats:
        mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
        print(f"  GPU {gpu['id']}:")
        print(f"    GPU利用率: {gpu['gpu_util']:5.1f}% {'█' * int(gpu['gpu_util']/5):<20}")
        print(f"    显存使用: {gpu['mem_used']:6.0f} / {gpu['mem_total']:6.0f} MiB ({mem_percent:5.1f}%)")
        print(f"    功耗: {gpu['power']:5.1f} W")

    # CPU信息
    print("\n【CPU状态】")
    print(f"  总CPU使用率: {cpu_stats['cpu_percent']:5.1f}% {'█' * int(cpu_stats['cpu_percent']/5):<20}")
    print(f"  内存使用率: {cpu_stats['memory_percent']:5.1f}%")

    if cpu_stats.get('python_procs'):
        print("\n  Python训练进程:")
        for proc in cpu_stats['python_procs'][:3]:  # 只显示前3个
            print(f"    PID {proc['pid']}: CPU {proc['cpu']:5.1f}%, 内存 {proc['memory']:5.1f}%")

    # 性能建议
    print("\n【性能建议】")
    if gpu_stats:
        avg_gpu_util = sum(g['gpu_util'] for g in gpu_stats) / len(gpu_stats)
        if avg_gpu_util < 50:
            print("  ⚠️  GPU利用率较低，可能的原因:")
            print("      - CPU-GPU数据传输瓶颈")
            print("      - batch size太小")
            print("      - 数据预处理占用CPU时间")
        elif avg_gpu_util > 80:
            print("  ✅ GPU利用率良好!")
        else:
            print("  ℹ️  GPU利用率中等，还有优化空间")

    if cpu_stats['cpu_percent'] > 80:
        print("  ⚠️  CPU占用过高，建议:")
        print("      - 增加batch size")
        print("      - 优化数据预处理")
        print("      - 使用pin_memory")

    print("="*80)

def main():
    """主循环"""
    print("开始监控训练性能...")
    print("按 Ctrl+C 停止\n")

    interval = 2.0  # 刷新间隔（秒）

    try:
        while True:
            gpu_stats = get_gpu_stats()
            cpu_stats = get_cpu_stats()

            print_stats(gpu_stats, cpu_stats)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == '__main__':
    main()
