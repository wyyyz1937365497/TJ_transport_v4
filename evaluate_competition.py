"""
比赛评分评估脚本

根据赛题评分标准计算模型得分：
- 最终得分: S_total = S_perf × P_int
- 性能得分: S_perf = S_efficiency × W_efficiency + S_stability × W_stability
- 干预成本惩罚因子: P_int ∈ (0, 1]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.env.competition_env import CompetitionSumoEnv
from src.models.traffic_controller import TrafficController


class CompetitionEvaluator:
    """比赛评分评估器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 比赛评分权重
        self.weights = {
            'efficiency': 0.85,      # 初赛以效率为主
            'stability': 0.15,       # 稳定性为辅
        }

        # 干预成本系数
        self.cost_params = {
            'alpha': config.get('competition', {}).get('intervention_cost', {}).get('alpha', 1.0),
            'beta': config.get('competition', {}).get('intervention_cost', {}).get('beta', 5.0),
            'cost_penalty_factor': config.get('competition', {}).get('intervention_cost', {}).get('cost_penalty_factor', 0.01),
        }

        # 控制比例（25% ICV）
        self.control_ratio = config.get('environment', {}).get('control_ratio', 0.25)

        print("=" * 80)
        print("🏆 智能交通协同控制竞赛 - 评分评估系统")
        print("=" * 80)
        print(f"设备: {self.device}")
        print(f"控制比例: {self.control_ratio * 100}%")
        print(f"效率权重: {self.weights['efficiency']}")
        print(f"稳定性权重: {self.weights['stability']}")

    def load_model(self, checkpoint_path: str) -> Any:
        """加载训练好的模型"""
        print(f"\n📦 加载模型: {checkpoint_path}")

        # 尝试多种可能的SB3模型路径
        sb3_paths = [
            checkpoint_path.replace('.pth', '_lagrangian.zip'),  # Phase 4 模型
            checkpoint_path.replace('.pth', '_sb3.zip'),         # 其他 SB3 模型
            checkpoint_path.replace('.pth', '.zip').replace('final_model', 'e2e_phase3_sb3'),  # Phase 3 模型
        ]

        # 尝试加载SB3模型
        for sb3_path in sb3_paths:
            if os.path.exists(sb3_path):
                try:
                    from stable_baselines3 import PPO
                    model = PPO.load(sb3_path, device=self.device)
                    print(f"✅ SB3模型加载成功: {sb3_path}")
                    return model
                except Exception as e:
                    print(f"⚠️ SB3模型加载失败 ({sb3_path}): {e}")
                    continue

        # 尝试加载PyTorch模型（兼容格式）
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # 检查是否是SB3兼容格式
                if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                    # 加载 SB3 policy
                    from src.models.sb3_full_policy import FullTrafficActorCriticPolicy
                    from stable_baselines3 import PPO
                    import gymnasium as gym

                    # 创建临时环境来初始化模型
                    temp_env = gym.make('CartPoleEnv')  # 临时环境，仅用于初始化

                    # 创建 PPO 模型并加载 policy state_dict
                    model = PPO(
                        FullTrafficActorCriticPolicy,
                        temp_env,
                        verbose=0,
                        device=self.device
                    )
                    model.policy.load_state_dict(checkpoint['policy_state_dict'])
                    print(f"✅ 兼容格式模型加载成功")
                    return model
                else:
                    # 尝试直接加载到 TrafficController
                    model = TrafficController(self.config.get('model', {}))
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    model.to(self.device)
                    model.eval()
                    print(f"✅ PyTorch模型加载成功")
                    return model
            except Exception as e:
                print(f"⚠️ 模型加载失败: {e}")

        print(f"⚠️ 未找到有效模型，将使用随机策略")
        return None

    def evaluate_episode(self, model: Any, use_gui: bool = False) -> Dict[str, Any]:
        """运行单个episode并收集统计数据"""
        env = CompetitionSumoEnv(self.env_config, use_gui=use_gui)

        # 数据收集
        all_speeds = []
        all_accelerations = []
        all_vehicles = set()  # 所有出现过的车辆
        arrived_vehicles = set()  # 到达的车辆
        vehicle_completion = {}  # 车辆完成度

        # 控制统计
        control_actions = []
        controlled_vehicles = set()

        observation = env.reset()
        step = 0
        max_steps = self.env_config.get('max_steps', 3600)

        print(f"\n🚗 开始仿真 (最大步数: {max_steps})...")

        while step < max_steps:
            # 准备batch
            batch = self._prepare_batch(observation, env)

            # 模型推理
            actions_to_apply = {}

            if model is not None:
                try:
                    # 调用模型
                    if hasattr(model, 'predict'):
                        # SB3模型：需要将字典观测转换为扁平化向量
                        flat_obs = self._format_observation(observation)
                        action, _ = model.predict(flat_obs, deterministic=True)

                        # 将扁平化动作转换为车辆控制字典
                        actions_to_apply = self._parse_actions(action, observation)
                    else:
                        # PyTorch模型
                        with torch.no_grad():
                            output = model(batch)
                            selected_vehicles = output.get('selected_vehicle_ids', [])
                            actions_list = output.get('safe_actions', [])

                            # 转换为字典格式
                            for i, veh_id in enumerate(selected_vehicles):
                                if veh_id in observation['vehicle_ids']:
                                    actions_to_apply[veh_id] = actions_list[i].cpu().numpy()

                except Exception as e:
                    print(f"⚠️ 模型推理失败: {e}")
                    actions_to_apply = {}

            # 记录控制统计
            for veh_id, action in actions_to_apply.items():
                control_actions.append({
                    'vehicle_id': veh_id,
                    'step': step,
                    'acceleration': float(action[0]),
                    'lane_change_prob': float(action[1]),
                })
                controlled_vehicles.add(veh_id)

            # 推进仿真
            observation, reward, done, info = env.step(actions_to_apply if actions_to_apply else None)

            # 收集统计
            current_vehicles = set(observation['vehicle_ids'])
            all_vehicles.update(current_vehicles)

            for veh_id in observation['vehicle_ids']:
                state = observation['vehicle_states'][veh_id]
                all_speeds.append(state['speed'])
                all_accelerations.append(abs(state['acceleration']))

                # 计算完成度（基于行驶距离）
                if veh_id not in vehicle_completion:
                    vehicle_completion[veh_id] = {
                        'distance': state.get('position', 0),
                        'departed': True
                    }
                else:
                    vehicle_completion[veh_id]['distance'] = max(
                        vehicle_completion[veh_id]['distance'],
                        state.get('position', 0)
                    )

            # 检查到达车辆
            if 'arrived' in info:
                for veh_id in info['arrived']:
                    arrived_vehicles.add(veh_id)
                    if veh_id in vehicle_completion:
                        vehicle_completion[veh_id]['arrived'] = True

            step += 1

            # 进度报告
            if step % 500 == 0:
                print(f"   步数 {step}/{max_steps} | 活跃车辆: {len(observation['vehicle_ids'])} | "
                      f"已到达: {len(arrived_vehicles)}")

            # 检查是否结束
            if done:
                break

        env.close()

        # 计算指标
        metrics = self._compute_metrics(
            all_speeds=all_speeds,
            all_accelerations=all_accelerations,
            all_vehicles=all_vehicles,
            arrived_vehicles=arrived_vehicles,
            vehicle_completion=vehicle_completion,
            control_actions=control_actions,
            controlled_vehicles=controlled_vehicles,
            total_steps=step
        )

        return metrics

    def _prepare_batch(self, observation: Dict, env) -> Dict[str, Any]:
        """准备推理batch"""
        vehicle_states = observation['vehicle_states']
        vehicle_ids = observation['vehicle_ids']
        icv_ids = observation['icv_ids']
        global_stats = observation['global_stats']

        batch = {
            'vehicle_states': vehicle_states,
            'vehicle_ids': vehicle_ids,
            'icv_ids': icv_ids,
            'global_metrics': torch.tensor(global_stats, dtype=torch.float32).unsqueeze(0).to(self.device),
            'is_icv': torch.tensor(
                [1 if vid in icv_ids else 0 for vid in vehicle_ids],
                dtype=torch.float32
            ).to(self.device) if vehicle_ids else torch.tensor([], dtype=torch.float32).to(self.device)
        }

        return batch

    def _format_observation(self, observation: Dict) -> np.ndarray:
        """
        将SUMO字典观测格式化为SB3所需的扁平化向量

        Args:
            observation: SUMO原始观测字典

        Returns:
            扁平化的观测数组 [MAX_VEHICLES * 9 + 32 + 1]
        """
        from src.constants import MAX_VEHICLES, FEATURES_PER_VEHICLE

        vehicle_states = observation.get('vehicle_states', {})
        global_stats = observation.get('global_stats', np.zeros(32))
        icv_ids = observation.get('icv_ids', set())

        # 车辆状态向量化（Frenet坐标系9维特征）
        vehicle_features = []
        for veh_id, state in vehicle_states.items():
            # 提取特征并归一化
            features = [
                state.get('s', 0.0) / 1000.0,              # 纵向位置
                state.get('d', 0.0) / 10.0,               # 横向偏移
                state.get('vs', 0.0) / 30.0,              # 纵向速度
                state.get('vd', 0.0) / 10.0,              # 横向速度
                state.get('speed', 0.0) / 30.0,           # 总速度
                state.get('acceleration', 0.0) / 3.0,     # 加速度
                state.get('lane_index', 0.0) / 10.0,      # 车道索引
                state.get('angle', 0.0) / 360.0,          # 航向角
                1.0 if veh_id in icv_ids else 0.0         # is_icv标志
            ]
            vehicle_features.extend(features)

        # Padding到固定大小
        max_features = MAX_VEHICLES * FEATURES_PER_VEHICLE
        if len(vehicle_features) < max_features:
            vehicle_features.extend([0.0] * (max_features - len(vehicle_features)))
        else:
            vehicle_features = vehicle_features[:max_features]

        # 拼接所有特征为扁平数组
        flat_obs = np.array(vehicle_features, dtype=np.float32)
        flat_obs = np.concatenate([
            flat_obs,                              # 车辆状态特征 (MAX_VEHICLES * 9)
            global_stats.flatten(),                # 全局统计特征 (32)
            [len(vehicle_states)]                  # 车辆数量 (1)
        ]).astype(np.float32)

        return flat_obs

    def _parse_actions(self, action: np.ndarray, observation: Dict) -> Dict[str, np.ndarray]:
        """
        解析SB3扁平化动作向量为车辆控制字典

        Args:
            action: 扁平化的 [MAX_VEHICLES * 2] 数组
            observation: 当前观测字典

        Returns:
            {vehicle_id: [acceleration, lane_change]}
        """
        from src.constants import MAX_VEHICLES

        actions = {}
        icv_ids = list(observation.get('icv_ids', set()))

        # 将扁平化的动作向量重新整形为 [max_vehicles, 2]
        action_reshaped = action.reshape(MAX_VEHICLES, 2)

        # 为每个ICV分配动作（只对ICV车辆进行控制）
        for i, veh_id in enumerate(icv_ids[:MAX_VEHICLES]):
            if i < len(action_reshaped):
                actions[veh_id] = action_reshaped[i]

        return actions

    def _compute_metrics(
        self,
        all_speeds: List[float],
        all_accelerations: List[float],
        all_vehicles: set,
        arrived_vehicles: set,
        vehicle_completion: Dict,
        control_actions: List[Dict],
        controlled_vehicles: set,
        total_steps: int
    ) -> Dict[str, Any]:
        """计算评估指标"""

        # ========== 1. OD完成率 (OCR) ==========
        total_vehicles = len(all_vehicles)
        arrived_count = len(arrived_vehicles)

        # 计算在途车辆的平均完成度
        in_road_completion_sum = 0
        in_road_count = 0
        for veh_id in all_vehicles:
            if veh_id not in arrived_vehicles and veh_id in vehicle_completion:
                # 简化：使用行驶距离作为完成度的近似
                # 实际应该对比路径总长度
                completion = min(vehicle_completion[veh_id].get('distance', 0) / 1000.0, 1.0)
                in_road_completion_sum += completion
                in_road_count += 1

        avg_in_road_completion = in_road_completion_sum / max(in_road_count, 1)

        # OCR = (已到达 + 在途完成度) / 总车辆数
        ocr = (arrived_count + in_road_completion_sum) / max(total_vehicles, 1)

        # ========== 2. 速度统计 ==========
        avg_speed = np.mean(all_speeds) if all_speeds else 0
        std_speed = np.std(all_speeds) if len(all_speeds) > 1 else 0

        # ========== 3. 加速度统计 ==========
        avg_abs_acceleration = np.mean(all_accelerations) if all_accelerations else 0

        # ========== 4. 干预成本 ==========
        # 总控制次数
        total_control_count = len(control_actions)

        # 控制车辆数
        num_controlled_vehicles = len(controlled_vehicles)

        # 加速度成本
        accel_costs = [abs(a['acceleration']) for a in control_actions]
        total_accel_cost = sum(accel_costs) if accel_costs else 0

        # 换道成本
        lane_changes = sum(1 for a in control_actions if a['lane_change_prob'] > 0.5)
        total_lane_change_cost = lane_changes * self.cost_params['beta']

        # 总干预成本
        total_intervention_cost = (
            self.cost_params['alpha'] * total_accel_cost +
            total_lane_change_cost
        )

        # 成本密度（每时间步的平均成本）
        cost_density = total_intervention_cost / max(total_steps, 1)

        # ========== 5. 计算得分 ==========

        # 效率得分（基于OCR，归一化到0-1）
        s_efficiency = ocr

        # 稳定性得分（速度越稳定越好）
        # 速度标准差越小越好，归一化：假设σv=10为最差，σv=0为最好
        s_stability = max(0, 1 - std_speed / 10.0)

        # 性能得分
        s_perf = (
            s_efficiency * self.weights['efficiency'] +
            s_stability * self.weights['stability']
        )

        # 干预成本惩罚因子
        # 成本越高，惩罚越大，P_int ∈ (0, 1]
        # 假设成本密度阈值
        max_acceptable_cost = 0.5
        p_int = max(0.1, 1 - (cost_density / max_acceptable_cost) * self.cost_params['cost_penalty_factor'])
        p_int = min(p_int, 1.0)

        # 最终得分
        s_total = s_perf * p_int

        return {
            'total_vehicles': total_vehicles,
            'arrived_vehicles': arrived_count,
            'in_road_vehicles': in_road_count,
            'avg_in_road_completion': avg_in_road_completion,
            'ocr': ocr,
            'avg_speed': avg_speed,
            'std_speed': std_speed,
            'avg_abs_acceleration': avg_abs_acceleration,
            'total_control_count': total_control_count,
            'num_controlled_vehicles': num_controlled_vehicles,
            'total_accel_cost': total_accel_cost,
            'total_lane_change_cost': total_lane_change_cost,
            'total_intervention_cost': total_intervention_cost,
            'cost_density': cost_density,
            's_efficiency': s_efficiency,
            's_stability': s_stability,
            's_perf': s_perf,
            'p_int': p_int,
            's_total': s_total,
            'total_steps': total_steps,
        }

    def evaluate(self, model: Any, num_episodes: int = 5, use_gui: bool = False) -> Dict[str, Any]:
        """运行多次评估并计算平均得分"""

        print(f"\n📊 开始评估 ({num_episodes} episodes)...")

        all_metrics = []

        for episode in range(num_episodes):
            print(f"\n{'='*80}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*80}")

            metrics = self.evaluate_episode(model, use_gui=use_gui)
            all_metrics.append(metrics)

            # 打印单个episode结果
            self._print_episode_results(metrics, episode + 1)

        # 计算平均和标准差
        final_results = self._aggregate_metrics(all_metrics)

        # 打印最终结果
        self._print_final_results(final_results, num_episodes)

        return final_results

    def _print_episode_results(self, metrics: Dict, episode_num: int):
        """打印单个episode结果"""
        print(f"\n📈 Episode {episode_num} 评估结果:")
        print(f"{'─'*80}")
        print(f"📊 基础指标:")
        print(f"   总车辆数:        {metrics['total_vehicles']}")
        print(f"   已到达:          {metrics['arrived_vehicles']}")
        print(f"   在途车辆:        {metrics['in_road_vehicles']}")
        print(f"   OD完成率(OCR):   {metrics['ocr']:.4f}")
        print(f"\n📊 性能指标:")
        print(f"   平均速度:        {metrics['avg_speed']:.2f} m/s")
        print(f"   速度标准差(σv):  {metrics['std_speed']:.2f} m/s")
        print(f"   平均绝对加速度:  {metrics['avg_abs_acceleration']:.3f} m/s²")
        print(f"\n📊 干预成本:")
        print(f"   控制指令总数:    {metrics['total_control_count']}")
        print(f"   受控车辆数:      {metrics['num_controlled_vehicles']}")
        print(f"   加速度成本:      {metrics['total_accel_cost']:.2f}")
        print(f"   换道成本:        {metrics['total_lane_change_cost']:.2f}")
        print(f"   总干预成本:      {metrics['total_intervention_cost']:.2f}")
        print(f"   成本密度:        {metrics['cost_density']:.4f}")
        print(f"\n🏆 得分:")
        print(f"   效率得分(Seff):  {metrics['s_efficiency']:.4f}")
        print(f"   稳定性得分(Sst): {metrics['s_stability']:.4f}")
        print(f"   性能得分(Sperf): {metrics['s_perf']:.4f}")
        print(f"   成本惩罚(Pint):  {metrics['p_int']:.4f}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   🥇 最终得分:      {metrics['s_total']:.4f}")

    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """聚合多次episode的指标"""

        aggregated = {}
        metric_keys = all_metrics[0].keys()

        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values) if len(values) > 1 else 0

        # 使用均值作为最终得分
        aggregated['final_score'] = aggregated['s_total_mean']
        aggregated['final_score_std'] = aggregated['s_total_std']

        return aggregated

    def _print_final_results(self, results: Dict, num_episodes: int):
        """打印最终评估结果"""
        print(f"\n\n{'='*80}")
        print(f"🏆 最终评估结果 ({num_episodes} episodes 平均)")
        print(f"{'='*80}")

        print(f"\n📊 性能指标:")
        print(f"   OD完成率(OCR):       {results['ocr_mean']:.4f} ± {results['ocr_std']:.4f}")
        print(f"   平均速度:            {results['avg_speed_mean']:.2f} ± {results['avg_speed_std']:.2f} m/s")
        print(f"   速度标准差(σv):      {results['std_speed_mean']:.2f} ± {results['std_speed_std']:.2f} m/s")
        print(f"   平均绝对加速度:      {results['avg_abs_acceleration_mean']:.3f} ± {results['avg_abs_acceleration_std']:.3f} m/s²")

        print(f"\n📊 干预成本:")
        print(f"   平均控制指令数:      {results['total_control_count_mean']:.0f}")
        print(f"   平均受控车辆数:      {results['num_controlled_vehicles_mean']:.0f}")
        print(f"   平均总干预成本:      {results['total_intervention_cost_mean']:.2f}")

        print(f"\n🏆 比赛得分:")
        print(f"   效率得分(Seff):      {results['s_efficiency_mean']:.4f} ± {results['s_efficiency_std']:.4f}")
        print(f"   稳定性得分(Sst):     {results['s_stability_mean']:.4f} ± {results['s_stability_std']:.4f}")
        print(f"   性能得分(Sperf):     {results['s_perf_mean']:.4f} ± {results['s_perf_std']:.4f}")
        print(f"   成本惩罚(Pint):      {results['p_int_mean']:.4f} ± {results['p_int_std']:.4f}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   🥇 最终得分:          {results['final_score']:.4f} ± {results['final_score_std']:.4f}")
        print(f"{'='*80}\n")


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='智能交通协同控制竞赛 - 评分评估')
    parser.add_argument('--config', type=str, default='configs/competition_preliminary.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径（默认使用final_model.pth）')
    parser.add_argument('--episodes', type=int, default=5,
                       help='评估episodes数量')
    parser.add_argument('--gui', action='store_true',
                       help='启用SUMO GUI可视化')
    parser.add_argument('--output', type=str, default=None,
                       help='结果输出文件路径（JSON格式）')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 确定检查点路径
    if args.checkpoint is None:
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/competition')
        args.checkpoint = os.path.join(checkpoint_dir, 'final_model.pth')

    # 创建评估器
    evaluator = CompetitionEvaluator(config)

    # 加载模型
    model = evaluator.load_model(args.checkpoint)

    # 运行评估
    results = evaluator.evaluate(model, num_episodes=args.episodes, use_gui=args.gui)

    # 保存结果
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存到: {args.output}")

    return results


if __name__ == '__main__':
    main()
