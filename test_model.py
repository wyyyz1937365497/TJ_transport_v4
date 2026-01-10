"""
模型测试脚本
验证神经网络组件是否可以正常初始化和运行
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np

print("="*70)
print("🧪 神经网络组件测试")
print("="*70)

# 测试1: 导入模块
print("\n1️⃣  测试模块导入...")
try:
    from src.models import create_model_from_config
    print("   ✅ 模块导入成功")
except Exception as e:
    print(f"   ❌ 模块导入失败: {e}")
    sys.exit(1)

# 测试2: 创建模型
print("\n2️⃣  测试模型创建...")
config = {
    "device": "cpu",
    "node_dim": 9,
    "edge_dim": 4,
    "gnn_hidden_dim": 64,
    "gnn_output_dim": 256,
    "gnn_layers": 3,
    "gnn_heads": 4,
    "gnn_dropout": 0.1,
    "world_hidden_dim": 128,
    "future_steps": 5,
    "world_dropout": 0.1,
    "controller_hidden_dim": 128,
    "global_dim": 16,
    "action_dim": 2,
    "top_k": 5,
    "controller_dropout": 0.2,
    "interaction_radius": 100.0,
    "max_neighbors": 8,
    "lane_change_distance": 50.0,
    "ttc_threshold": 2.0,
    "thw_threshold": 1.5,
    "max_accel": 2.0,
    "max_decel": -3.0,
    "emergency_decel": -5.0,
    "max_lane_change_speed": 5.0,
    "cost_limit": 0.1,
    "initial_lambda": 1.0,
    "lambda_lr": 0.01
}

try:
    model = create_model_from_config(config)
    print("   ✅ 模型创建成功")
except Exception as e:
    print(f"   ❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: GNN前向传播
print("\n3️⃣  测试GNN前向传播...")
try:
    # 创建模拟图数据
    num_nodes = 10
    num_edges = 15

    node_features = torch.randn(num_nodes, 9)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_features = torch.randn(num_edges, 4)

    # 测试GNN
    gnn_output = model.risk_gnn(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        batch=None
    )

    print(f"   ✅ GNN前向传播成功")
    print(f"      - 节点嵌入形状: {gnn_output['node_embedding'].shape}")
    print(f"      - 全局嵌入形状: {gnn_output['global_embedding'].shape}")

except Exception as e:
    print(f"   ❌ GNN前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 世界模型
print("\n4️⃣  测试世界模型...")
try:
    gnn_embedding = gnn_output['node_embedding']
    world_output = model.world_model(gnn_embedding)

    print(f"   ✅ 世界模型前向传播成功")
    if 'next_state' in world_output:
        print(f"      - 下一状态形状: {world_output['next_state'].shape}")

except Exception as e:
    print(f"   ❌ 世界模型前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 控制器
print("\n5️⃣  测试控制器...")
try:
    batch_size = num_nodes
    global_metrics = torch.randn(1, 16)
    vehicle_ids = [f"veh_{i}" for i in range(num_nodes)]
    is_icv = torch.randint(0, 2, (num_nodes,)).float()

    # 准备输入 - 使用world_model生成真实预测
    # 通过world_model进行前向传播获取预测
    try:
        world_output = model.world_model(gnn_embedding)
        world_predictions = world_output  # 使用真实的world_model输出
    except:
        # 如果world_model需要特殊输入格式，使用gnn_embedding
        world_predictions = gnn_embedding

    controller_output = model.controller(
        gnn_embedding=gnn_embedding,
        world_predictions=world_predictions,
        global_metrics=global_metrics,
        vehicle_ids=vehicle_ids,
        is_icv=is_icv
    )

    print(f"   ✅ 控制器前向传播成功")
    print(f"      - 选中车辆数: {len(controller_output['selected_vehicle_ids'])}")
    print(f"      - 动作形状: {controller_output['raw_actions'].shape}")

except Exception as e:
    print(f"   ❌ 控制器前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 完整模型
print("\n6️⃣  测试完整模型...")
try:
    # 准备完整batch
    batch = {
        'vehicle_states': {},
        'vehicle_ids': vehicle_ids,
        'icv_ids': set([vehicle_ids[i] for i in range(num_nodes) if is_icv[i] > 0.5]),
        'global_metrics': global_metrics,
        'is_icv': is_icv
    }

    # 构建图
    from src.models import GraphBuilder
    graph_builder = GraphBuilder()
    graph_data = graph_builder.build_graph(batch['vehicle_states'], batch['icv_ids'])

    # 如果图是空的，创建一个虚拟图
    if graph_data.x.size(0) == 0:
        graph_data.x = torch.randn(num_nodes, 9)
        graph_data.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        graph_data.edge_attr = torch.randn(num_edges, 4)

    batch['graph_data'] = graph_data

    # 运行完整模型
    output = model(batch)

    print(f"   ✅ 完整模型前向传播成功")
    print(f"      - 选中车辆: {len(output['selected_vehicle_ids'])}")

except Exception as e:
    print(f"   ❌ 完整模型前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ 所有测试通过!")
print("="*70)

print("\n💡 模型组件工作正常，可以开始训练了")
print("   运行命令:")
print("   - python quick_start.py")
print("   - python train.py")
