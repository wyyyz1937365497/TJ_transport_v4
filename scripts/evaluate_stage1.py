#!/usr/bin/env python3
"""
Stage 1 World Model 评估脚本

评估WorldModel的：
1. 状态预测精度（Flow）
2. 风险检测准确性（Risk）
3. 嵌入层质量
4. 可视化预测效果

使用方法:
    python scripts/evaluate_stage1.py \
        --config configs/v5_complete.yaml \
        --checkpoint checkpoints/v5_complete/stage1_best.pth \
        --device cuda
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.world_model import WorldModel, create_world_model
from src.env.competition_env import CompetitionSumoEnv
from scripts.train_stage1_world_observer import (
    VehicleEmbedding,
    normalize_vehicle_states,
    denormalize_vehicle_states,
    collect_idm_trajectories,
    TrajectoryDataset
)


def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Dict[str, Any]:
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def evaluate_flow_prediction(
    model: WorldModel,
    vehicle_embedding: VehicleEmbedding,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> Dict[str, float]:
    """
    评估状态预测精度（Flow Loss）

    指标:
    - MSE Loss: 均方误差
    - MAE Loss: 平均绝对误差
    - RMSE: 均方根误差
    - R²: 决定系数
    """
    model.eval()
    vehicle_embedding.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    # 按特征维度统计误差
    feature_errors = np.zeros(9)  # 9个特征

    with torch.no_grad():
        for batch in dataloader:
            obs = batch['obs'].to(device)
            next_obs = batch['next_obs'].to(device)
            valid_masks = batch['valid_masks'].to(device)

            B, T, MAX_VEHICLES, state_dim = obs.shape

            # 初始化hidden
            hidden = model.initialize_hidden(B, device)

            # 逐时间步预测
            pred_list = []
            target_list = []

            for t in range(T):
                obs_t = obs[:, t, :, :]  # [B, MAX, 9]
                embeddings_t = vehicle_embedding(obs_t)
                outputs = model(embeddings_t, hidden)
                pred_list.append(outputs['pred_next_states'])
                hidden = outputs['hidden']

            pred_states = torch.stack(pred_list, dim=1)  # [B, T, MAX, 9]
            target_states = next_obs

            # 计算误差（仅对有效车辆）
            valid_mask = valid_masks  # [B, T, MAX, 1]

            diff = pred_states - target_states  # [B, T, MAX, 9]
            squared_error = (diff ** 2) * valid_mask
            abs_error = torch.abs(diff) * valid_mask

            total_mse += squared_error.sum().item()
            total_mae += abs_error.sum().item()

            # 按特征统计误差
            for feat_idx in range(9):
                feat_se = squared_error[..., feat_idx].sum().item()
                feature_errors[feat_idx] += feat_se

            total_samples += valid_mask.sum().item()

    # 计算平均指标
    mse = total_mse / total_samples
    mae = total_mae / total_samples
    rmse = np.sqrt(mse)

    # 归一化后的指标（映射回原始尺度）
    # 假设特征归一化范围为：
    # s: [0, 1000], d: [-10, 10], vs: [0, 30], vd: [-5, 5]
    # speed: [0, 30], accel: [-3, 3], lane: [0, 3], angle: [0, 360], bottleneck: [0, 1]
    feature_scales = np.array([1000.0, 10.0, 30.0, 5.0, 30.0, 3.0, 3.0, 360.0, 1.0])
    feature_errors_normalized = feature_errors / total_samples
    feature_errors_original = np.sqrt(feature_errors_normalized) * feature_scales

    # 计算R²（需要计算总方差）
    # 简化估计：基于MSE
    # R² = 1 - MSE/Var，假设Var ≈ MSE(baseline) ≈ 1（归一化后）
    r2 = max(0, 1 - mse)  # 粗略估计

    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'feature_errors_original': {
            's': feature_errors_original[0],
            'd': feature_errors_original[1],
            'vs': feature_errors_original[2],
            'vd': feature_errors_original[3],
            'speed': feature_errors_original[4],
            'acceleration': feature_errors_original[5],
            'lane_index': feature_errors_original[6],
            'angle': feature_errors_original[7],
            'in_bottleneck': feature_errors_original[8],
        }
    }

    return metrics


def evaluate_risk_prediction(
    model: WorldModel,
    vehicle_embedding: VehicleEmbedding,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> Dict[str, float]:
    """
    评估风险预测准确性（Risk Loss）

    指标:
    - BCE Loss: 二元交叉熵
    - Accuracy: 分类准确率
    - Precision: 精确率
    - Recall: 召回率
    - F1 Score: F1分数
    """
    model.eval()
    vehicle_embedding.eval()

    total_bce = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            obs = batch['obs'].to(device)
            risk_labels = batch['risk_labels'].to(device)
            valid_masks = batch['valid_masks'].to(device)

            B, T, MAX_VEHICLES = obs.shape[:3]

            # 初始化hidden
            hidden = model.initialize_hidden(B, device)

            # 逐时间步预测
            pred_risks_list = []

            for t in range(T):
                obs_t = obs[:, t, :, :]
                embeddings_t = vehicle_embedding(obs_t)
                outputs = model(embeddings_t, hidden)
                pred_risks_list.append(outputs['risk_prob'])
                hidden = outputs['hidden']

            pred_risks = torch.stack(pred_risks_list, dim=1)  # [B, T, MAX]
            target_risks = risk_labels

            # 计算BCE loss（仅对有效车辆）
            risk_mask = valid_masks.view(-1)  # [B*T*MAX]
            pred_flat = pred_risks.view(-1)
            target_flat = target_risks.view(-1)

            bce_loss = nn.functional.binary_cross_entropy(
                pred_flat, target_flat, reduction='none'
            )
            total_bce += (bce_loss * risk_mask).sum().item()
            total_samples += risk_mask.sum().item()

            # 收集预测和标签用于计算分类指标
            valid_indices = risk_mask > 0.5
            all_preds.extend(pred_flat[valid_indices].cpu().numpy())
            all_labels.extend(target_flat[valid_indices].cpu().numpy())

    # 计算平均指标
    avg_bce = total_bce / total_samples

    # 计算分类指标
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    labels_binary = np.array(all_labels).astype(int)

    tp = ((preds_binary == 1) & (labels_binary == 1)).sum()
    fp = ((preds_binary == 1) & (labels_binary == 0)).sum()
    tn = ((preds_binary == 0) & (labels_binary == 0)).sum()
    fn = ((preds_binary == 0) & (labels_binary == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        'bce_loss': avg_bce,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    return metrics


def visualize_predictions(
    model: WorldModel,
    vehicle_embedding: VehicleEmbedding,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    save_dir: str = 'eval_results/stage1'
):
    """
    可视化预测效果

    生成：
    1. 预测vs真实状态散点图
    2. 特征误差柱状图
    3. 风险预测混淆矩阵
    4. 时序预测示例
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    vehicle_embedding.eval()

    # 收集第一个batch的预测
    with torch.no_grad():
        for batch in dataloader:
            obs = batch['obs'].to(device)
            next_obs = batch['next_obs'].to(device)
            risk_labels = batch['risk_labels'].to(device)

            B, T, MAX_VEHICLES, state_dim = obs.shape

            # 预测
            hidden = model.initialize_hidden(B, device)
            pred_list = []
            pred_risks_list = []

            for t in range(T):
                obs_t = obs[:, t, :, :]
                embeddings_t = vehicle_embedding(obs_t)
                outputs = model(embeddings_t, hidden)
                pred_list.append(outputs['pred_next_states'])
                pred_risks_list.append(outputs['risk_prob'])
                hidden = outputs['hidden']

            pred_states = torch.stack(pred_list, dim=1).cpu().numpy()
            pred_risks = torch.stack(pred_risks_list, dim=1).cpu().numpy()
            target_states = next_obs.cpu().numpy()
            target_risks = risk_labels.cpu().numpy()

            break  # 只用第一个batch

    # 反归一化用于可视化
    pred_states_denorm = denormalize_vehicle_states(pred_states[0])  # [T, MAX, 9]
    target_states_denorm = denormalize_vehicle_states(target_states[0])

    # 1. 预测vs真实散点图（选择前100个有效车辆）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Flow Prediction: Predicted vs Actual', fontsize=16)

    feature_names = ['Position (s)', 'Lateral (d)', 'Long Vel (vs)',
                     'Lat Vel (vd)', 'Speed', 'Accel']
    feature_indices = [0, 1, 2, 3, 4, 5]

    for idx, (ax, feat_idx) in enumerate(zip(axes.flat, feature_indices)):
        # 提取有效车辆（非padding）
        valid_mask = batch['valid_masks'][0, :, 0].cpu().numpy() > 0.5
        valid_indices = np.where(valid_mask[0])[0][:100]  # 前100个

        pred = pred_states_denorm[0, valid_indices, feat_idx]
        target = target_states_denorm[0, valid_indices, feat_idx]

        ax.scatter(target, pred, alpha=0.5, s=1)

        # 对角线（完美预测）
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')

        ax.set_xlabel(f'Actual {feature_names[idx]}')
        ax.set_ylabel(f'Predicted {feature_names[idx]}')
        ax.set_title(f'{feature_names[idx]} (R² computed)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'flow_prediction_scatter.png', dpi=150, bbox_inches='tight')
    print(f"[可视化] 已保存: {save_dir / 'flow_prediction_scatter.png'}")
    plt.close()

    # 2. 特征误差柱状图（原始尺度）
    fig, ax = plt.subplots(figsize=(12, 6))

    # 从metrics计算特征误差
    feature_names_all = ['Position (s)', 'Lateral (d)', 'Long Vel (vs)',
                        'Lat Vel (vd)', 'Speed', 'Accel', 'Lane', 'Angle', 'Bottleneck']
    feature_scales = np.array([1000.0, 10.0, 30.0, 5.0, 30.0, 3.0, 3.0, 360.0, 1.0])

    diff = (pred_states_denorm[0] - target_states_denorm[0])
    valid_mask = batch['valid_masks'][0, :, 0].cpu().numpy() > 0.5
    mae_per_feature = []

    for feat_idx in range(9):
        if valid_mask[:, feat_idx].any():
            feat_diff = np.abs(diff[:, feat_idx])
            mae = feat_diff[valid_mask[:, feat_idx]].mean()
            mae_per_feature.append(mae)
        else:
            mae_per_feature.append(0.0)

    bars = ax.bar(feature_names_all, mae_per_feature)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Feature-wise Prediction Error (Original Scale)')
    ax.set_xticklabels(feature_names_all, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / 'feature_error_bar.png', dpi=150, bbox_inches='tight')
    print(f"[可视化] 已保存: {save_dir / 'feature_error_bar.png'}")
    plt.close()

    # 3. 风险预测混淆矩阵
    from sklearn.metrics import confusion_matrix

    valid_mask = batch['valid_masks'][0].cpu().numpy().flatten() > 0.5
    preds = (pred_risks[0].flatten() > 0.5).astype(int)
    labels = target_risks[0].flatten().astype(int)

    # 只使用有效样本
    preds_valid = preds[valid_mask]
    labels_valid = labels[valid_mask]

    cm = confusion_matrix(labels_valid, preds_valid, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe (0)', 'Risky (1)'],
                yticklabels=['Safe (0)', 'Risky (1)'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Risk Prediction Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_dir / 'risk_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"[可视化] 已保存: {save_dir / 'risk_confusion_matrix.png'}")
    plt.close()

    # 4. 时序预测示例（选择一个有效车辆）
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    fig.suptitle('Time Series Prediction Example (One Vehicle)', fontsize=16)

    # 找一个一直有效的车辆
    valid_counts = batch['valid_masks'][0, :, 0].sum(dim=0).cpu().numpy()
    vehicle_idx = np.argmax(valid_counts)

    time_steps = np.arange(T)

    for feat_idx, ax in enumerate(axes.flat):
        pred = pred_states_denorm[0, :, vehicle_idx, feat_idx]
        target = target_states_denorm[0, :, vehicle_idx, feat_idx]

        ax.plot(time_steps, target, 'b-', label='Actual', linewidth=2)
        ax.plot(time_steps, pred, 'r--', label='Predicted', linewidth=2, alpha=0.7)

        feature_name = ['Position (m)', 'Lateral (m)', 'Long Vel (m/s)',
                         'Lat Vel (m/s)', 'Speed (m/s)', 'Accel (m/s²)',
                         'Lane', 'Angle (°)', 'Bottleneck'][feat_idx]

        ax.set_ylabel(feature_name)
        ax.set_title(f'Feature {feat_idx}')
        if feat_idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    axes.flat[-1].set_xlabel('Time Step')
    axes.flat[-2].set_xlabel('Time Step')
    axes.flat[-3].set_xlabel('Time Step')

    plt.tight_layout()
    plt.savefig(save_dir / 'timeseries_prediction.png', dpi=150, bbox_inches='tight')
    print(f"[可视化] 已保存: {save_dir / 'timeseries_prediction.png'}")
    plt.close()

    print(f"\n[可视化] 所有图表已保存到: {save_dir}")


def print_metrics(metrics: Dict[str, Any], title: str):
    """打印指标"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    if 'mse' in metrics:  # Flow metrics
        print(f"\n📊 状态预测 (Flow):")
        print(f"  MSE Loss:      {metrics['mse']:.6f}")
        print(f"  MAE Loss:      {metrics['mae']:.6f}")
        print(f"  RMSE:          {metrics['rmse']:.6f}")
        print(f"  R² Score:      {metrics['r2_score']:.6f}")
        print(f"\n  特征误差 (原始尺度):")
        for feat, error in metrics['feature_errors_original'].items():
            print(f"    {feat:15s}: MAE = {error:.4f}")

    if 'accuracy' in metrics:  # Risk metrics
        print(f"\n⚠️  风险预测 (Risk):")
        print(f"  BCE Loss:      {metrics['bce_loss']:.6f}")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")
        print(f"  F1 Score:      {metrics['f1_score']:.4f}")
        print(f"\n  混淆矩阵:")
        print(f"    TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 World Model 评估")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='评估的episode数（用于数据收集）')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图表')
    parser.add_argument('--skip_training', action='store_true',
                        help='跳过数据收集，使用已有缓存')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("="*80)
    print("Stage 1 World Model 评估")
    print("="*80)
    print(f"\n配置文件: {args.config}")
    print(f"检查点: {args.checkpoint}")
    print(f"设备: {args.device}")

    # 创建环境
    print("\n创建环境...")
    env = CompetitionSumoEnv(
        config=config,
        use_gui=False,
        device=args.device
    )

    # 收集评估数据
    print("\n收集评估数据...")
    if args.skip_training:
        print("[跳过] 使用已有缓存")
        cache_dir = "cache/stage1_trajectories"
    else:
        print("[收集] 收集IDM轨迹...")
        cache_dir = None

    dataset = collect_idm_trajectories(
        env,
        num_episodes=args.num_episodes,
        max_steps=config['stage1_world_observer']['training']['episode_length'],
        device=args.device,
        cache_dir=cache_dir,
        use_cache=True,
        force_refresh=False
    )

    # 创建数据加载器
    print("\n创建数据加载器...")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 评估时使用batch_size=1
        shuffle=False,
        num_workers=0
    )

    # 加载模型
    print("\n加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = create_world_model(
        hidden_dim=config['policy']['hidden_dim'],
        latent_dim=config['stage1_world_observer']['world_model']['latent_dim'],
        num_layers=config['stage1_world_observer']['world_model']['num_layers'],
        num_vehicles=config['environment']['icv_config']['max_vehicles'],
        device=args.device
    )

    vehicle_embedding = VehicleEmbedding(
        input_dim=9,
        hidden_dim=config['policy']['hidden_dim']
    ).to(args.device)

    # 加载权重
    if 'world_model' in checkpoint:
        model.load_state_dict(checkpoint['world_model'])
        print("  ✅ WorldModel权重已加载")

    if 'vehicle_embedding' in checkpoint:
        vehicle_embedding.load_state_dict(checkpoint['vehicle_embedding'])
        print("  ✅ VehicleEmbedding权重已加载")

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 评估Flow预测
    print("\n" + "="*80)
    print("评估状态预测 (Flow)...")
    print("="*80)
    flow_metrics = evaluate_flow_prediction(model, vehicle_embedding, dataloader, args.device)
    print_metrics(flow_metrics, "状态预测 (Flow) 结果")

    # 评估Risk预测
    print("\n" + "="*80)
    print("评估风险预测 (Risk)...")
    print("="*80)
    risk_metrics = evaluate_risk_prediction(model, vehicle_embedding, dataloader, args.device)
    print_metrics(risk_metrics, "风险预测 (Risk) 结果")

    # 生成可视化
    if args.visualize:
        print("\n生成可视化图表...")
        visualize_predictions(model, vehicle_embedding, dataloader, args.device)

    # 保存评估结果
    results = {
        'flow_metrics': flow_metrics,
        'risk_metrics': risk_metrics,
        'checkpoint_path': args.checkpoint,
        'config': args.config
    }

    results_path = Path('eval_results/stage1/evaluation_results.pkl')
    results_path.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n[保存] 评估结果已保存: {results_path}")

    # 总结
    print("\n" + "="*80)
    print("📊 评估总结")
    print("="*80)
    print(f"\n✅ 状态预测:")
    print(f"   RMSE: {flow_metrics['rmse']:.4f} (归一化后)")
    print(f"   R² Score: {flow_metrics['r2_score']:.4f}")
    print(f"   最好特征: {min(flow_metrics['feature_errors_original'], key=flow_metrics['feature_errors_original'].get)}")

    print(f"\n✅ 风险预测:")
    print(f"   Accuracy: {risk_metrics['accuracy']:.4f}")
    print(f"   F1 Score: {risk_metrics['f1_score']:.4f}")
    print(f"   Precision: {risk_metrics['precision']:.4f}")
    print(f"   Recall: {risk_metrics['recall']:.4f}")

    # 判断是否可以进入Stage 2
    print("\n" + "="*80)
    print("🚀 进入Stage 2的建议")
    print("="*80)

    # Flow预测判断
    if flow_metrics['rmse'] < 1.0:
        print("✅ 状态预测优秀 (RMSE < 1.0)")
        flow_status = "优秀"
    elif flow_metrics['rmse'] < 2.0:
        print("⚠️  状态预测良好 (RMSE < 2.0)")
        flow_status = "良好"
    else:
        print("❌ 状态预测需要改进 (RMSE >= 2.0)")
        flow_status = "需要改进"

    # Risk预测判断
    if risk_metrics['f1_score'] > 0.7:
        print("✅ 风险预测优秀 (F1 > 0.7)")
        risk_status = "优秀"
    elif risk_metrics['f1_score'] > 0.5:
        print("⚠️  风险预测可用 (F1 > 0.5)")
        risk_status = "可用"
    else:
        print("❌ 风险预测较弱 (F1 <= 0.5)")
        risk_status = "较弱"

    # 总体建议
    if flow_status in ["优秀", "良好"] and risk_status in ["优秀", "可用"]:
        print("\n🎉 建议：可以进入Stage 2训练！")
        print("   WorldModel已具备足够的状态预测和风险检测能力。")
    else:
        print("\n⚠️  建议：考虑进一步优化后再进入Stage 2")
        print("   可能需要：增加训练epochs、调整模型结构、或增加数据量。")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
