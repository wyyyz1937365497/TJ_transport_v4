"""
训练瓶颈诊断脚本 - 在训练更新阶段运行

使用方法：
    python diagnose_bottleneck.py

这个脚本会分析：
1. GPU内存和使用率
2. CPU使用率
3. 内存占用
4. 可能的瓶颈点
"""

import subprocess
import time
import psutil
import torch

def get_gpu_stats():
    """获取GPU统计信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,utilization.memory',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'gpu_util': int(parts[2]),
                        'mem_used': int(parts[3]),
                        'mem_total': int(parts[4]),
                        'mem_util': int(parts[5])
                    })
            return gpu_info
    except Exception as e:
        print(f"[ERROR] 无法获取GPU信息: {e}")

    return None

def get_cpu_stats():
    """获取CPU统计信息"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

        # 获取进程列表（按CPU使用率排序）
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 0:  # 只显示有CPU使用的进程
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 按CPU使用率排序，取top 10
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        top_processes = processes[:10]

        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'load_avg': load_avg,
            'top_processes': top_processes
        }
    except Exception as e:
        print(f"[ERROR] 无法获取CPU信息: {e}")
        return None

def get_memory_stats():
    """获取内存统计信息"""
    try:
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent
        }
    except Exception as e:
        print(f"[ERROR] 无法获取内存信息: {e}")
        return None

def analyze_bottleneck(gpu_stats, cpu_stats, mem_stats):
    """分析瓶颈"""
    print("\n" + "=" * 80)
    print("瓶颈分析")
    print("=" * 80)

    bottlenecks = []

    # GPU分析
    if gpu_stats:
        for gpu in gpu_stats:
            print(f"\nGPU {gpu['index']} ({gpu['name']}):")
            print(f"  GPU利用率: {gpu['gpu_util']}%")
            print(f"  显存使用: {gpu['mem_used']}MB / {gpu['mem_total']}MB ({gpu['mem_util']}%)")

            if gpu['gpu_util'] < 50:
                print(f"  ⚠️  GPU利用率低 ({gpu['gpu_util']}%) - 可能是CPU-GPU传输瓶颈")
                bottlenecks.append('GPU利用率低')

            if gpu['mem_util'] < 50:
                print(f"  ⚠️  显存使用率低 ({gpu['mem_util']}%) - batch_size可能太小")
                bottlenecks.append('显存未充分利用')

    # CPU分析
    if cpu_stats:
        print(f"\nCPU:")
        print(f"  总体使用率: {cpu_stats['cpu_percent']}%")
        print(f"  核心数: {cpu_stats['cpu_count']}")
        print(f"  负载平均: {cpu_stats['load_avg'][0]:.2f} (1分钟)")

        if cpu_stats['cpu_percent'] > 80:
            print(f"  ⚠️  CPU使用率高 ({cpu_stats['cpu_percent']}%) - 可能是CPU瓶颈")
            bottlenecks.append('CPU瓶颈')

        print(f"\nTop 10 CPU使用进程:")
        for i, proc in enumerate(cpu_stats['top_processes'], 1):
            print(f"  {i:2d}. PID {proc['pid']:6d} | {proc['name']:20s} | CPU: {proc['cpu_percent']:5.1f}% | MEM: {proc['memory_percent']:5.1f}%")

    # 内存分析
    if mem_stats:
        print(f"\n内存:")
        print(f"  使用: {mem_stats['used_gb']:.1f}GB / {mem_stats['total_gb']:.1f}GB ({mem_stats['percent']}%)")
        print(f"  可用: {mem_stats['available_gb']:.1f}GB")

        if mem_stats['percent'] > 90:
            print(f"  ⚠️  内存使用率高 ({mem_stats['percent']}%) - 可能需要减少batch_size")
            bottlenecks.append('内存瓶颈')

    # 总结
    print("\n" + "=" * 80)
    print("瓶颈总结")
    print("=" * 80)

    if not bottlenecks:
        print("✅ 未发现明显瓶颈")
    else:
        print("发现的瓶颈:")
        for i, b in enumerate(bottlenecks, 1):
            print(f"  {i}. {b}")

    # 建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)

    if gpu_stats and any(g['gpu_util'] < 50 for g in gpu_stats):
        print("- GPU利用率低:")
        print("  1. 检查是否有大量CPU-GPU数据传输")
        print("  2. 增大batch_size以提高GPU并行度")
        print("  3. 使用profiler找出CPU-bound操作")

    if cpu_stats and cpu_stats['cpu_percent'] > 80:
        print("- CPU使用率高:")
        print("  1. 数据预处理可能成为瓶颈")
        print("  2. 环境step可能占用大量CPU（SUMO仿真）")
        print("  3. 考虑使用多进程并行环境")

    if mem_stats and mem_stats['percent'] > 90:
        print("- 内存使用率高:")
        print("  1. 减少num_envs或batch_size")
        print("  2. 检查是否有内存泄漏")

    print("\n")

if __name__ == '__main__':
    print("=" * 80)
    print("训练瓶颈诊断")
    print("=" * 80)
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 收集统计信息
    gpu_stats = get_gpu_stats()
    cpu_stats = get_cpu_stats()
    mem_stats = get_memory_stats()

    # 分析瓶颈
    analyze_bottleneck(gpu_stats, cpu_stats, mem_stats)
