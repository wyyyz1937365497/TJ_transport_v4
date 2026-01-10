"""
测试多GPU训练
验证多GPU设置和超参数调整是否正常工作
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn

def test_multi_gpu():
    """测试多GPU设置"""
    print("="*70)
    print("🧪 多GPU训练测试")
    print("="*70)

    # 1. 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法测试多GPU")
        return False

    device_count = torch.cuda.device_count()
    print(f"\n✅ 检测到 {device_count} 张CUDA设备")

    # 2. 打印GPU信息
    from src.utils import print_gpu_info
    print_gpu_info()

    # 3. 创建简单模型
    print("\n🏗️  创建测试模型...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()

    # 4. 设置多GPU
    print("\n🚀 设置多GPU训练...")
    from src.utils import setup_multi_gpu_training, adjust_hyperparameters_for_multi_gpu, monitor_gpu_usage

    model = setup_multi_gpu_training(model)

    # 5. 测试超参数调整
    print("\n📊 测试超参数调整...")

    if device_count > 1:
        # 原始配置
        original_config = {
            'batch_size': 64,
            'lr': 1e-4
        }

        print(f"原始配置:")
        print(f"  - Batch size: {original_config['batch_size']}")
        print(f"  - 学习率: {original_config['lr']:.6f}")

        # 调整后的配置
        adjusted_config = adjust_hyperparameters_for_multi_gpu(original_config, device_count)

        print(f"\n调整后配置 (x{device_count}):")
        print(f"  - Batch size: {adjusted_config['batch_size']}")
        print(f"  - 学习率: {adjusted_config['lr']:.6f}")

        # 计算理论加速比
        speedup = device_count * 0.85  # DataParallel通常有85%效率
        print(f"\n💡 理论加速比: {speedup:.2f}x (考虑通信开销)")

    # 6. 测试前向传播
    print("\n🧪 测试模型推理...")

    batch_size = 128
    x = torch.randn(batch_size, 256)

    if device_count > 1:
        x = x.cuda()

    with torch.no_grad():
        output = model(x)

    print(f"✅ 模型推理成功")
    print(f"   - 输入形状: {x.shape}")
    print(f"   - 输出形状: {output.shape}")

    # 7. 监控GPU使用
    print("\n📊 GPU使用情况:")
    monitor_gpu_usage()

    # 8. 测试DataParallel行为
    if hasattr(model, 'device_ids'):
        print(f"\n✅ DataParallel已启用")
        print(f"   - 设备列表: {model.device_ids}")
        print(f"   - 输出设备: {model.output_device}")
    else:
        print(f"\n⚠️  DataParallel未启用（单GPU模式）")

    print("\n" + "="*70)
    print("✅ 多GPU测试完成!")
    print("="*70)

    return True


def test_training_speed():
    """测试训练速度"""
    print("\n" + "="*70)
    print("⚡ 训练速度测试")
    print("="*70)

    import time

    # 创建模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = TestModel()

    # 设置多GPU
    from src.utils import setup_multi_gpu_training, monitor_gpu_usage
    model = setup_multi_gpu_training(model)

    # 训练参数
    batch_size = 128
    num_iterations = 100
    num_gpus = torch.cuda.device_count()

    if torch.cuda.is_available():
        batch_size = batch_size * max(num_gpus, 1)
        print(f"\n⚙️  测试配置:")
        print(f"   - GPU数量: {num_gpus}")
        print(f"   - Batch size: {batch_size} (x{num_gpus})")
        print(f"   - 迭代次数: {num_iterations}")

    # 创建数据
    x = torch.randn(batch_size, 256)
    y = torch.randn(batch_size, 256)

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 训练循环
    print(f"\n🏋️  开始训练...")

    start_time = time.time()

    for i in range(num_iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            it_per_sec = (i + 1) / elapsed
            print(f"   迭代 {i+1}/{num_iterations} | 耗时: {elapsed:.1f}s | "
                  f"速度: {it_per_sec:.1f} it/s")

    total_time = time.time() - start_time

    print(f"\n✅ 训练完成!")
    print(f"   - 总耗时: {total_time:.2f}s")
    print(f"   - 平均速度: {num_iterations/total_time:.1f} iterations/s")

    # 显示GPU使用
    monitor_gpu_usage()

    return True


if __name__ == "__main__":
    try:
        # 测试1: 基本多GPU设置
        success1 = test_multi_gpu()

        # 测试2: 训练速度
        success2 = test_training_speed()

        if success1 and success2:
            print(f"\n🎉 所有测试通过!")
            print(f"\n💡 现在可以开始多GPU训练了:")
            print(f"   python train.py --phase 1")
        else:
            print(f"\n❌ 部分测试失败")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
