"""
并行数据收集器 - 真正的多进程实现
使用 Python multiprocessing 加速 SUMO 数据收集
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import pickle
import json
from multiprocessing import Pool, Process, Queue, Manager, cpu_count
import multiprocessing

# 尝试设置spawn启动方法（避免fork导致的TraCI状态共享问题）
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 已经设置过，忽略
    pass

from .sumo_env import SumoEnvironment


def collect_single_episode(args: Tuple[Dict[str, Any], int, int, float, int]) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """
    单个 episode 的数据收集（在独立进程中运行）

    Args:
        args: (config, episode_id, max_steps, timeout, port)

    Returns:
        (trajectories, stats)
    """
    config, episode_id, max_steps, timeout, port = args

    # 独立进程环境 - 确保TraCI状态干净
    import sys
    import traci

    # 清理任何现有的TraCI连接
    try:
        traci.close()
    except:
        pass

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

    trajectories = {}
    stats = {
        'episode_id': episode_id,
        'total_steps': 0,
        'total_vehicles': 0,
        'collection_time': 0.0,
        'success': False,
        'error': None
    }

    try:
        # 创建环境（每个进程独立实例，使用指定端口，禁用端口重试）
        env = SumoEnvironment(config, use_gui=False, port=port, disable_port_retry=True)

        start_time = time.time()
        observation = env.reset()

        # 数据收集循环
        for step in range(max_steps):
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                break

            # 记录数据
            vehicle_states = observation['vehicle_states']
            current_time = step * config.get('step_length', 0.1)

            for veh_id, state in vehicle_states.items():
                if veh_id not in trajectories:
                    trajectories[veh_id] = {
                        'id': veh_id,
                        'timestamps': [],
                        'positions': [],
                        'speeds': [],
                        'accelerations': [],
                        'lane_ids': [],
                        'lanes': []
                    }

                trajectories[veh_id]['timestamps'].append(current_time)
                trajectories[veh_id]['positions'].append(state['position'])
                trajectories[veh_id]['speeds'].append(state['speed'])
                trajectories[veh_id]['accelerations'].append(state['acceleration'])
                trajectories[veh_id]['lane_ids'].append(state['lane_id'])
                trajectories[veh_id]['lanes'].append(state['lane_index'])

            # 推进仿真
            observation, reward, done, info = env.step(actions=None)

            # 检查是否结束
            if len(observation['vehicle_states']) == 0 and step > 100:
                break

            if done:
                break

        # 关闭环境
        try:
            env.close()
        except:
            pass

        # 统计
        stats['total_steps'] = step + 1
        stats['total_vehicles'] = len(trajectories)
        stats['collection_time'] = time.time() - start_time
        stats['success'] = True

    except Exception as e:
        stats['error'] = str(e)
        stats['success'] = False

    return trajectories, stats


class ParallelDataCollector:
    """
    并行数据收集器

    特性：
    - 使用 multiprocessing.Pool 实现真正的并行
    - 每个 SUMO 实例在独立进程中运行
    - 自动负载均衡
    - 进度监控
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_workers: Optional[int] = None,
        timeout: float = 180.0
    ):
        """
        Args:
            config: SUMO 配置
            num_workers: 工作进程数（默认使用 CPU 核心数 - 1）
            timeout: 每个 episode 的超时时间（秒）
        """
        self.config = config
        self.timeout = timeout

        # 确定工作进程数
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)  # 保留一个核心

        self.num_workers = min(num_workers, 8)  # 最多 8 个并行 SUMO

        print(f"📊 并行数据收集器初始化")
        print(f"   - 工作进程数: {self.num_workers}")
        print(f"   - 超时时间: {timeout}s")
        print(f"   - CPU 核心数: {cpu_count()}")

    def collect_episodes(
        self,
        num_episodes: int,
        max_steps: int = 3600,
        verbose: bool = True
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """
        并行收集多个 episodes 的数据

        Args:
            num_episodes: episode 数量
            max_steps: 每个 episode 的最大步数
            verbose: 是否打印进度

        Returns:
            (all_trajectories, total_stats)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🚀 开始并行数据收集")
            print(f"{'='*70}")
            print(f"   - Episodes: {num_episodes}")
            print(f"   - 并行进程: {self.num_workers}")
            print(f"   - 最大步数: {max_steps}")
            print(f"{'='*70}\n")

        start_time = time.time()

        # 准备任务参数 - 为每个episode分配不同端口
        tasks = []
        base_port = 8813
        for i in range(num_episodes):
            # 每个episode使用不同的端口，间隔10个端口
            port = base_port + i * 10
            tasks.append((self.config, i, max_steps, self.timeout, port))

        # 使用进程池并行执行
        all_trajectories = {}
        all_stats = []

        # 使用spawn启动方式避免TraCI状态共享问题
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=self.num_workers) as pool:
            # 异步应用函数
            results = pool.map_async(collect_single_episode, tasks)

            # 监控进度
            if verbose:
                completed = 0
                while not results.ready():
                    new_completed = len([s for s in all_stats if 'episode_id' in s])
                    if new_completed > completed:
                        completed = new_completed
                        elapsed = time.time() - start_time
                        print(f"   进度: {completed}/{num_episodes} episodes | "
                              f"耗时: {elapsed:.1f}s | "
                              f"速度: {completed/max(elapsed, 0.1):.2f} ep/s")

                    time.sleep(0.5)  # 避免过度轮询

            # 获取所有结果
            results_list = results.get()

            # 处理结果
            for trajectories, stats in results_list:
                if stats['success']:
                    all_trajectories.update(trajectories)
                    if verbose:
                        print(f"   ✅ Episode {stats['episode_id']}: "
                              f"{stats['total_vehicles']} 辆车, "
                              f"{stats['total_steps']} 步, "
                              f"{stats['collection_time']:.1f}s")
                else:
                    if verbose:
                        print(f"   ❌ Episode {stats['episode_id']}: 失败 - {stats['error']}")

                all_stats.append(stats)

        # 计算总体统计
        total_time = time.time() - start_time
        successful_episodes = [s for s in all_stats if s['success']]

        total_stats = {
            'total_episodes': num_episodes,
            'successful_episodes': len(successful_episodes),
            'failed_episodes': num_episodes - len(successful_episodes),
            'total_steps': sum(s['total_steps'] for s in successful_episodes),
            'total_vehicles': len(all_trajectories),
            'collection_time': total_time,
            'avg_time_per_episode': total_time / num_episodes if num_episodes > 0 else 0,
            'avg_steps_per_episode': sum(s['total_steps'] for s in successful_episodes) / max(len(successful_episodes), 1),
            'throughput': num_episodes / total_time if total_time > 0 else 0
        }

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ 并行数据收集完成!")
            print(f"{'='*70}")
            print(f"   - 总 episodes: {total_stats['total_episodes']}")
            print(f"   - 成功: {total_stats['successful_episodes']}")
            print(f"   - 失败: {total_stats['failed_episodes']}")
            print(f"   - 总车辆数: {total_stats['total_vehicles']}")
            print(f"   - 总步数: {total_stats['total_steps']:,}")
            print(f"   - 总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
            print(f"   - 平均时间/episode: {total_stats['avg_time_per_episode']:.1f}s")
            print(f"   - 吞吐量: {total_stats['throughput']:.2f} episodes/s")
            print(f"   - 加速比: ~{self.num_workers}x (理论值)")
            print(f"{'='*70}\n")

        return all_trajectories, total_stats


def collect_parallel_data_optimized(
    config: Dict[str, Any],
    num_episodes: int = 10,
    max_steps: int = 3600,
    timeout: float = 180.0,
    num_workers: Optional[int] = None,
    output_dir: str = "data"
) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """
    优化的并行数据收集函数

    Args:
        config: SUMO 配置
        num_episodes: episode 数量
        max_steps: 最大步数
        timeout: 超时时间
        num_workers: 工作进程数
        output_dir: 输出目录

    Returns:
        (trajectories, stats)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建并行收集器
    collector = ParallelDataCollector(
        config=config,
        num_workers=num_workers,
        timeout=timeout
    )

    # 收集数据
    trajectories, stats = collector.collect_episodes(
        num_episodes=num_episodes,
        max_steps=max_steps,
        verbose=True
    )

    # 保存数据
    if len(trajectories) > 0:
        timestamp = int(time.time())
        filepath = os.path.join(output_dir, f'parallel_data_{timestamp}.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump({
                'trajectories': trajectories,
                'stats': stats
            }, f)

        # 保存统计
        stats_file = filepath.replace('.pkl', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"💾 数据已保存: {filepath}")

    return trajectories, stats
