"""
理想架构v4.0评估脚本

核心改进：
1. 使用完整的影响力驱动控制器进行推理
2. 显示Top-K选择和影响力评分
3. 支持动态权重可视化
4. 详细的性能分析和干预成本跟踪
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.env.competition_env import CompetitionSumoEnv
from stable_baselines3 import PPO


class IdealArchitectureEvaluator:
    """
    理想架构v4.0评估器

    核心特性：
    1. 使用完整的影响力驱动控制器
    2. Top-K稀疏控制
    3. 详细的性能分析
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = config.get('environment', {})
        self.device = torch.device(config.get('device', 'cuda'))

        # 从配置获取参数
        model_config = config.get('model', {})
        self.top_k = model_config.get('controller', {}).get('top_k', 5)
        self.max_vehicles = self.env_config.get('max_vehicles', 32)

    def load_model(self, checkpoint_path: str) -> Any:
        """加载训练好的模型"""
        print(f"\n📦 加载v4.0模型: {checkpoint_path}")

        # 尝试多种可能的路径
        possible_paths = [
            checkpoint_path,
            checkpoint_path.replace('.pth', '.zip'),
            checkpoint_path.replace('.zip', '.pth'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # 加载SB3模型
                    model = PPO.load(path, device=self.device)
                    print(f"[OK] 模型加载成功: {path}")
                    print(f"   Top-K: {self.top_k}")
                    return model
                except Exception as e:
                    print(f"[WARNING] 加载失败 ({path}): {e}")
                    continue

        print(f"[WARNING] 未找到有效模型，将使用随机策略")
        return None

    def evaluate_episode(self, model: Any, use_gui: bool = False) -> Dict[str, Any]:
        """运行单个episode"""
        env = CompetitionSumoEnv(self.env_config, use_gui=use_gui)

        # 数据收集
        all_speeds = []
        all_accelerations = []
        all_vehicles = set()
        arrived_vehicles = set()
        vehicle_completion = {}

        control_actions = []
        controlled_vehicles = set()

        # 新增：影响力评分跟踪
        influence_scores_history = []
        top_k_selections_history = []
        dynamic_weights_history = []

        observation = env.reset()
        step = 0
        max_steps = self.env_config.get('max_steps', 3600)

        print(f"\n🚗 开始仿真 (最大步数: {max_steps})...")

        while step < max_steps:
            # 使用v4.0完整推理逻辑（影响力驱动控制器）
            actions_to_apply, inference_info = self._infer_with_influence_controller(
                observation, model
            )

            # 记录推理信息
            if inference_info:
                if 'influence_scores' in inference_info:
                    influence_scores_history.append(inference_info['influence_scores'])
                if 'selected_vehicle_ids' in inference_info:
                    top_k_selections_history.append(inference_info['selected_vehicle_ids'])
                if 'dynamic_weights' in inference_info:
                    dynamic_weights_history.append(inference_info['dynamic_weights'])

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

                if veh_id not in vehicle_completion:
                    vehicle_completion[veh_id] = {
                        'distance': state.get('s', 0),
                        'departed': True
                    }
                else:
                    vehicle_completion[veh_id]['distance'] = max(
                        vehicle_completion[veh_id]['distance'],
                        state.get('s', 0)
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
                num_controlled = len(top_k_selections_history[-1]) if top_k_selections_history else 0
                print(f"   步数 {step}/{max_steps} | 活跃车辆: {len(observation['vehicle_ids'])} | "
                      f"已控制: {num_controlled} | 已到达: {len(arrived_vehicles)}")

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
            total_steps=step,
            influence_scores_history=influence_scores_history,
            top_k_selections_history=top_k_selections_history,
            dynamic_weights_history=dynamic_weights_history
        )

        return metrics

    def _infer_with_influence_controller(
        self,
        observation: Dict,
        model: Any
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        使用完整的影响力驱动控制器进行推理

        Returns:
            actions_to_apply: {vehicle_id: [acceleration, lane_change]}
            inference_info: 包含Top-K选择、影响力评分等信息
        """
        actions_to_apply = {}
        inference_info = {}

        if model is None:
            return actions_to_apply, inference_info

        try:
            # 准备观测
            flat_obs = self._format_observation(observation)
            obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 使用模型的predict方法（它会调用完整的影响力驱动控制器）
            with torch.no_grad():
                # 调用policy的predict方法
                actions, info = model.policy.predict(
                    observation=flat_obs,
                    deterministic=True,
                    return_top_k=True
                )

                # 将动作转换为车辆控制字典
                # 动作格式: [32*2] = [accel_0, ..., accel_31, lane_0, ..., lane_31]
                accel_actions = actions[:32]
                lane_actions = actions[32:64]

                # 获取ICV列表
                icv_ids = list(observation.get('icv_ids', set()))

                # 获取选中的车辆
                selected_indices = info.get('selected_indices', []) if info else []
                selected_vehicle_ids = info.get('selected_vehicle_ids', []) if info else []

                # 为选中的车辆分配动作
                for i, veh_id in enumerate(selected_vehicle_ids):
                    if i < len(selected_indices) and i < 32:
                        local_idx = selected_indices[i]

                        accel = accel_actions[local_idx]
                        lane = lane_actions[local_idx]

                        # 映射到实际范围
                        accel_mapped = accel * 2.5  # [-1, 1] -> [-2.5, 2.5]
                        accel_mapped = np.clip(accel_mapped, -3.0, 2.0)

                        actions_to_apply[veh_id] = np.array([accel_mapped, lane])

                # 保存推理信息
                inference_info = {
                    'selected_vehicle_ids': selected_vehicle_ids,
                    'selected_indices': selected_indices,
                    'influence_scores': info.get('influence_scores', np.zeros(32)),
                    'dynamic_weights': info.get('dynamic_weights', np.zeros(3)),
                    'z_flow': info.get('z_flow', 0.0),
                    'z_risk': info.get('z_risk', 0.0),
                }

        except Exception as e:
            print(f"[WARNING] 推理失败: {e}")
            import traceback
            traceback.print_exc()

        return actions_to_apply, inference_info

    def _format_observation(self, observation: Dict) -> np.ndarray:
        """格式化观测为扁平化向量"""
        vehicle_states = observation['vehicle_states']
        global_stats = observation.get('global_stats', np.zeros(32))
        icv_ids = observation.get('icv_ids', set())

        # 车辆特征向量化
        vehicle_features = []
        for veh_id, state in vehicle_states.items():
            features = [
                state.get('s', 0.0) / 1000.0,
                state.get('d', 0.0) / 10.0,
                state.get('vs', 0.0) / 30.0,
                state.get('vd', 0.0) / 10.0,
                state.get('speed', 0.0) / 30.0,
                state.get('acceleration', 0.0) / 3.0,
                state.get('lane_index', 0.0) / 10.0,
                state.get('angle', 0.0) / 360.0,
                1.0 if veh_id in icv_ids else 0.0
            ]
            vehicle_features.extend(features)

        # Padding
        max_features = 32 * 9
        if len(vehicle_features) < max_features:
            vehicle_features.extend([0.0] * (max_features - len(vehicle_features)))
        else:
            vehicle_features = vehicle_features[:max_features]

        # 拼接
        flat_obs = np.array(vehicle_features, dtype=np.float32)
        flat_obs = np.concatenate([
            flat_obs,
            global_stats.flatten(),
            [len(vehicle_states)]
        ])

        return flat_obs

    def _compute_metrics(
        self,
        all_speeds: List[float],
        all_accelerations: List[float],
        all_vehicles: set,
        arrived_vehicles: set,
        vehicle_completion: Dict,
        control_actions: List[Dict],
        controlled_vehicles: set,
        total_steps: int,
        influence_scores_history: List[np.ndarray] = None,
        top_k_selections_history: List[List] = None,
        dynamic_weights_history: List[np.ndarray] = None
    ) -> Dict[str, Any]:
        """计算评估指标"""
        total_vehicles = len(all_vehicles)
        arrived_count = len(arrived_vehicles)

        # 在途车辆完成度
        in_road_completion_sum = 0
        in_road_count = 0
        for veh_id in all_vehicles:
            if veh_id not in arrived_vehicles and veh_id in vehicle_completion:
                completion = min(vehicle_completion[veh_id].get('distance', 0) / 1000.0, 1.0)
                in_road_completion_sum += completion
                in_road_count += 1

        avg_in_road_completion = in_road_completion_sum / max(in_road_count, 1)

        # OD完成率
        ocr = (arrived_count + in_road_completion_sum) / max(total_vehicles, 1)

        # 性能指标
        avg_speed = np.mean(all_speeds) if all_speeds else 0.0
        std_speed = np.std(all_speeds) if len(all_speeds) > 1 else 0.0
        avg_abs_acceleration = np.mean(all_accelerations) if all_accelerations else 0.0

        # 控制统计
        total_control_count = len(control_actions)
        num_controlled_vehicles = len(controlled_vehicles)

        total_accel_cost = sum(abs(a['acceleration']) for a in control_actions)
        total_lane_change_cost = sum(a['lane_change_prob'] for a in control_actions)
        total_intervention_cost = total_accel_cost + total_lane_change_cost

        cost_density = total_intervention_cost / max(total_steps, 1)

        # 得分计算
        s_efficiency = ocr
        s_stability = max(0, 1 - std_speed / 10.0)
        s_perf = (s_efficiency * 0.85 + s_stability * 0.15)

        max_acceptable_cost = 0.5
        p_int = max(0.1, 1 - (cost_density / max_acceptable_cost) * 0.5)
        p_int = min(p_int, 1.0)

        s_total = s_perf * p_int

        # 影响力分析
        avg_top_k_selected = 0
        avg_influence_score = 0
        avg_dynamic_weights = np.zeros(3)

        if top_k_selections_history:
            avg_top_k_selected = np.mean([len(sel) for sel in top_k_selections_history])

        if influence_scores_history:
            all_scores = np.concatenate(influence_scores_history)
            avg_influence_score = np.mean(all_scores)

        if dynamic_weights_history:
            avg_dynamic_weights = np.mean(dynamic_weights_history, axis=0)

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
            # 新增：影响力分析
            'avg_top_k_selected': avg_top_k_selected,
            'avg_influence_score': avg_influence_score,
            'avg_dynamic_weights_efficiency': avg_dynamic_weights[0],
            'avg_dynamic_weights_stability': avg_dynamic_weights[1],
            'avg_dynamic_weights_cost': avg_dynamic_weights[2],
        }

    def _print_episode_results(self, metrics: Dict, episode_num: int):
        """打印单个episode结果"""
        print(f"\n📈 Episode {episode_num} 评估结果:")
        print(f"{'─'*80}")
        print(f"[INFO] 基础指标:")
        print(f"   总车辆数:        {metrics['total_vehicles']}")
        print(f"   已到达:          {metrics['arrived_vehicles']}")
        print(f"   在途车辆:        {metrics['in_road_vehicles']}")
        print(f"   OD完成率(OCR):   {metrics['ocr']:.4f}")
        print(f"\n[INFO] 性能指标:")
        print(f"   平均速度:        {metrics['avg_speed']:.2f} m/s")
        print(f"   速度标准差(σv):  {metrics['std_speed']:.2f} m/s")
        print(f"   平均绝对加速度:  {metrics['avg_abs_acceleration']:.3f} m/s²")
        print(f"\n[INFO] 干预成本:")
        print(f"   控制指令总数:    {metrics['total_control_count']}")
        print(f"   受控车辆数:      {metrics['num_controlled_vehicles']}")
        print(f"   平均Top-K选择:   {metrics['avg_top_k_selected']:.1f}")
        print(f"   加速度成本:      {metrics['total_accel_cost']:.2f}")
        print(f"   换道成本:        {metrics['total_lane_change_cost']:.2f}")
        print(f"   总干预成本:      {metrics['total_intervention_cost']:.2f}")
        print(f"   成本密度:        {metrics['cost_density']:.4f}")
        print(f"\n[INFO] 影响力分析:")
        print(f"   平均影响力得分:  {metrics['avg_influence_score']:.4f}")
        print(f"   动态权重 (效率):  {metrics['avg_dynamic_weights_efficiency']:.3f}")
        print(f"   动态权重 (稳定):  {metrics['avg_dynamic_weights_stability']:.3f}")
        print(f"   动态权重 (成本):  {metrics['avg_dynamic_weights_cost']:.3f}")
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

        aggregated['final_score'] = aggregated['s_total_mean']
        aggregated['final_score_std'] = aggregated['s_total_std']

        return aggregated

    def _print_final_results(self, results: Dict, num_episodes: int):
        """打印最终结果"""
        print(f"\n{'='*80}")
        print(f"🏆 最终评估结果 ({num_episodes} episodes 平均)")
        print(f"{'='*80}")
        print(f"\n[INFO] 基础指标:")
        print(f"   总车辆数:        {results['total_vehicles_mean']:.1f} ± {results['total_vehicles_std']:.1f}")
        print(f"   已到达:          {results['arrived_vehicles_mean']:.1f} ± {results['arrived_vehicles_std']:.1f}")
        print(f"   OD完成率(OCR):   {results['ocr_mean']:.4f} ± {results['ocr_std']:.4f}")
        print(f"\n[INFO] 性能指标:")
        print(f"   平均速度:        {results['avg_speed_mean']:.2f} ± {results['avg_speed_std']:.2f} m/s")
        print(f"   速度标准差:      {results['std_speed_mean']:.2f} ± {results['std_speed_std']:.2f} m/s")
        print(f"   平均绝对加速度:  {results['avg_abs_acceleration_mean']:.3f} ± {results['avg_abs_acceleration_std']:.3f} m/s²")
        print(f"\n[INFO] 干预成本:")
        print(f"   控制指令总数:    {results['total_control_count_mean']:.0f}")
        print(f"   受控车辆数:      {results['num_controlled_vehicles_mean']:.1f}")
        print(f"   平均Top-K选择:   {results['avg_top_k_selected_mean']:.1f}")
        print(f"   总干预成本:      {results['total_intervention_cost_mean']:.2f}")
        print(f"   成本密度:        {results['cost_density_mean']:.4f}")
        print(f"\n[INFO] 影响力分析:")
        print(f"   平均影响力得分:  {results['avg_influence_score_mean']:.4f}")
        print(f"   动态权重 (效率):  {results['avg_dynamic_weights_efficiency_mean']:.3f}")
        print(f"   动态权重 (稳定):  {results['avg_dynamic_weights_stability_mean']:.3f}")
        print(f"   动态权重 (成本):  {results['avg_dynamic_weights_cost_mean']:.3f}")
        print(f"\n🏆 最终得分:")
        print(f"   效率得分:        {results['s_efficiency_mean']:.4f}")
        print(f"   稳定性得分:      {results['s_stability_mean']:.4f}")
        print(f"   性能得分:        {results['s_perf_mean']:.4f}")
        print(f"   成本惩罚:        {results['p_int_mean']:.4f}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   🥇 总分:          {results['final_score']:.4f} ± {results['final_score_std']:.4f}")
        print(f"{'='*80}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="理想架构v4.0评估")
    parser.add_argument('--config', type=str, default='configs/competition_preliminary.yaml',
                        help='配置文件')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/v4_phase3/ideal_v4_final.zip',
                        help='模型checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='评估episodes')
    parser.add_argument('--gui', action='store_true', help='显示GUI')

    args = parser.parse_args()

    # 加载配置
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建评估器
    evaluator = IdealArchitectureEvaluator(config)

    # 加载模型
    model = evaluator.load_model(args.checkpoint)

    # 评估
    print(f"\n[INFO] 开始评估 ({args.episodes} episodes)...")

    all_metrics = []
    for episode in range(args.episodes):
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"{'='*80}")

        metrics = evaluator.evaluate_episode(model, use_gui=args.gui)
        all_metrics.append(metrics)

        evaluator._print_episode_results(metrics, episode + 1)

    # 最终结果
    final_results = evaluator._aggregate_metrics(all_metrics)
    evaluator._print_final_results(final_results, args.episodes)

    return final_results


if __name__ == '__main__':
    main()
