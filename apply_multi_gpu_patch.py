"""
自动应用多GPU和进度条支持的补丁脚本

运行此脚本会自动修改train.py以集成多GPU训练和tqdm进度条
"""

import re
import sys
from pathlib import Path


def apply_patch():
    """应用补丁到train.py"""

    train_py = Path(__file__).parent / "train.py"

    if not train_py.exists():
        print(f"[ERROR] {train_py} not found!")
        return False

    print(f"[INFO] Reading {train_py}...")
    with open(train_py, 'r', encoding='utf-8') as f:
        content = f.read()

    # ============================================================================
    # 补丁1: 修改Phase1WorldModelTrainer.__init__
    # ============================================================================

    # 找到__init__方法并添加multi_gpu支持
    old_init = '''    def __init__(self, config: Dict[str, Any], enhanced_manager: EnhancedTrainingManager):
        self.config = config
        self.enhanced_manager = enhanced_manager
        self.device = get_device()

        # 多GPU配置
        self.use_cuda = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count()
        self.multi_gpu = self.num_gpus > 1

        # Checkpoint目录
        base_dir = config.get('paths', {}).get('checkpoint_dir', 'checkpoints')
        self.checkpoint_dir = os.path.join(base_dir, 'phase1')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 配置
        phase1_config = config.get('training', {}).get('phase1', {})
        self.num_episodes = phase1_config.get('num_episodes', 50)
        self.epochs = phase1_config.get('epochs', 30)
        self.batch_size = phase1_config.get('batch_size', 256)
        self.learning_rate = phase1_config.get('learning_rate', 1e-4)
        self.num_workers = phase1_config.get('num_parallel_workers', 4)

        # 数据缓存
        self.cache_dir = os.path.join(self.checkpoint_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'data.pkl')

        # 打印配置
        print(f"\\n[PHASE 1] Configuration:")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Parallel Workers: {self.num_workers}")
        print(f"  [GPU] CUDA Available: {self.use_cuda}")
        print(f"  [GPU] GPU Count: {self.num_gpus}")
        print(f"  [GPU] Multi-GPU Training: {self.multi_gpu}")
        print(f"  [OK] Curriculum Learning: ENABLED (default)")
        print(f"     Levels: {len(enhanced_manager.curriculum.levels)}")'''

    new_init = '''    def __init__(self, config: Dict[str, Any], enhanced_manager: EnhancedTrainingManager):
        self.config = config
        self.enhanced_manager = enhanced_manager

        # 初始化多GPU管理器
        self.gpu_manager = MultiGPUManager(config)
        self.device = self.gpu_manager.device

        # 初始化训练指标记录器
        self.metrics = TrainingMetrics()

        # Checkpoint目录
        base_dir = config.get('paths', {}).get('checkpoint_dir', 'checkpoints')
        self.checkpoint_dir = os.path.join(base_dir, 'phase1')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 配置
        phase1_config = config.get('training', {}).get('phase1', {})
        self.num_episodes = phase1_config.get('num_episodes', 50)
        self.epochs = phase1_config.get('epochs', 30)
        self.batch_size = phase1_config.get('batch_size', 256)
        self.learning_rate = phase1_config.get('learning_rate', 1e-4)
        self.num_workers = phase1_config.get('num_parallel_workers', 4)

        # 数据缓存
        self.cache_dir = os.path.join(self.checkpoint_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'data.pkl')

        # 打印配置
        print(f"\\n[PHASE 1] Configuration:")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Parallel Workers: {self.num_workers}")
        print(f"  Effective Batch Size: {self.gpu_manager.get_effective_batch_size(self.batch_size)}")
        print(f"  [OK] Curriculum Learning: ENABLED (default)")
        print(f"     Levels: {len(enhanced_manager.curriculum.levels)}")'''

    if old_init in content:
        content = content.replace(old_init, new_init)
        print("[OK] Patched Phase1WorldModelTrainer.__init__")
    else:
        print("[SKIP] Phase1WorldModelTrainer.__init__ not found or already patched")

    # ============================================================================
    # 补丁2: 修改train方法 - 添加模型包装
    # ============================================================================

    # 找到模型创建后添加包装
    old_model_creation = '''        model = IdealTrafficControllerV4(
            node_dim=model_config.get('gnn', {}).get('node_dim', 9),
            edge_dim=model_config.get('gnn', {}).get('edge_dim', 4),
            global_dim=model_config.get('controller', {}).get('global_dim', 32),
            gnn_hidden_dim=model_config.get('gnn', {}).get('hidden_dim', 64),
            gnn_output_dim=model_config.get('gnn', {}).get('output_dim', 256),
            rssm_hidden_dim=model_config.get('world_model', {}).get('hidden_dim', 128),
            rssm_latent_dim=model_config.get('world_model', {}).get('latent_dim', 64),
            controller_hidden_dim=model_config.get('controller', {}).get('hidden_dim', 128),
            top_k=model_config.get('controller', {}).get('top_k', 5),
            device=str(self.device)
        ).to(self.device)

        # 2. 收集数据（带课程学习）'''

    new_model_creation = '''        model = IdealTrafficControllerV4(
            node_dim=model_config.get('gnn', {}).get('node_dim', 9),
            edge_dim=model_config.get('gnn', {}).get('edge_dim', 4),
            global_dim=model_config.get('controller', {}).get('global_dim', 32),
            gnn_hidden_dim=model_config.get('gnn', {}).get('hidden_dim', 64),
            gnn_output_dim=model_config.get('gnn', {}).get('output_dim', 256),
            rssm_hidden_dim=model_config.get('world_model', {}).get('hidden_dim', 128),
            rssm_latent_dim=model_config.get('world_model', {}).get('latent_dim', 64),
            controller_hidden_dim=model_config.get('controller', {}).get('hidden_dim', 128),
            top_k=model_config.get('controller', {}).get('top_k', 5),
            device=str(self.device)
        ).to(self.device)

        # 多GPU包装
        model = self.gpu_manager.wrap_model(model)

        # 2. 收集数据（带课程学习和进度条）'''

    if old_model_creation in content:
        content = content.replace(old_model_creation, new_model_creation)
        print("[OK] Patched model creation in train()")
    else:
        print("[SKIP] Model creation not found or already patched")

    # ============================================================================
    # 补丁3: 修改数据收集以使用进度条
    # ============================================================================

    # 找到worker收集部分
    old_worker_collection = '''        observations = []
        next_observations = []

        collect_func = partial(_collect_worker, env_config=env_config, max_steps=500)

        with Pool(processes=num_workers) as pool:
            worker_results = []
            for worker_id, count in enumerate(episode_counts):
                if count > 0:
                    result = pool.apply_async(collect_func, (worker_id, count))
                    worker_results.append(result)

            completed = 0
            for result in worker_results:
                try:
                    worker_obs, worker_next_obs = result.get(timeout=600)
                    observations.extend(worker_obs)
                    next_observations.extend(worker_next_obs)
                    completed += 1
                    print(f"   Worker {completed}/{len(worker_results)} completed")
                except Exception as e:
                    print(f"   [WARNING] Worker failed: {e}")

        print(f"   [OK] Collected {len(observations)} transitions")'''

    new_worker_collection = '''        observations = []
        next_observations = []

        collect_func = partial(_collect_worker, env_config=env_config, max_steps=500)

        # 创建数据收集进度跟踪器
        collect_progress = DataCollectionProgress(num_workers)

        try:
            with Pool(processes=num_workers) as pool:
                worker_results = []
                for worker_id, count in enumerate(episode_counts):
                    if count > 0:
                        result = pool.apply_async(collect_func, (worker_id, count))
                        worker_results.append((worker_id, result))

                for worker_id, result in worker_results:
                    try:
                        worker_obs, worker_next_obs = result.get(timeout=600)
                        observations.extend(worker_obs)
                        next_observations.extend(worker_next_obs)

                        # 更新进度
                        collect_progress.update_worker(worker_id, len(worker_obs))

                    except Exception as e:
                        print(f"\\n   [WARNING] Worker {worker_id} failed: {e}")
        finally:
            collect_progress.close()

        print(f"   [OK] Collected {len(observations)} transitions")'''

    if old_worker_collection in content:
        content = content.replace(old_worker_collection, new_worker_collection)
        print("[OK] Patched data collection in _collect_level_data()")
    else:
        print("[SKIP] Data collection not found or already patched")

    # ============================================================================
    # 保存修改后的文件
    # ============================================================================

    backup_file = train_py.with_suffix('.py.backup')
    print(f"\\n[INFO] Creating backup: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        # 保存原始内容
        with open(train_py, 'r', encoding='utf-8') as original:
            f.write(original.read())

    print(f"[INFO] Writing patched file: {train_py}")
    with open(train_py, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\\n" + "="*60)
    print("[SUCCESS] Multi-GPU and progress bar support applied!")
    print("="*60)
    print("\\nNext steps:")
    print("1. Test the training: python train.py")
    print("2. Check GPU usage in training output")
    print("3. Monitor progress bars during training")
    print("\\nIf there are any issues, restore from backup:")
    print(f"  cp {backup_file} {train_py}")

    return True


if __name__ == "__main__":
    print("="*60)
    print("Multi-GPU Training Patch Tool")
    print("="*60)
    print()

    success = apply_patch()

    if success:
        print("\\n[OK] Patch applied successfully!")
        sys.exit(0)
    else:
        print("\\n[ERROR] Patch failed!")
        sys.exit(1)
