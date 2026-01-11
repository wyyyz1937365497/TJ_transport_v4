"""
全局常量配置文件

包含系统中的所有硬编码常量，便于统一管理和修改。
"""

# ============================================================================
# 环境配置常量
# ============================================================================

# SUMO 默认配置
DEFAULT_SUMO_PORT = 8813
DEFAULT_STEP_LENGTH = 0.1
DEFAULT_MAX_STEPS = 3600

# 车辆配置
MAX_VEHICLES = 32  # 最大支持的车辆数量
FEATURES_PER_VEHICLE = 5  # 每辆车的特征数: speed, acceleration, angle, lane_index, position

# 特征归一化参数（用于还原归一化的特征）
ANGLE_SCALE = 360.0  # 角度归一化因子
LANE_INDEX_SCALE = 10.0  # 车道索引归一化因子
POSITION_SCALE = 1000.0  # 位置归一化因子

# 图构建参数
DEFAULT_INTERACTION_RADIUS = 100.0  # 车辆交互半径（米）
DEFAULT_MAX_NEIGHBORS = 8  # 每辆车的最大邻居数
DEFAULT_LANE_CHANGE_DISTANCE = 50.0  # 换道距离阈值（米）

# ============================================================================
# 网络架构配置常量
# ============================================================================

# GNN 配置
GNN_NODE_DIM = 9  # 节点特征维度
GNN_EDGE_DIM = 4  # 边特征维度
GNN_HIDDEN_DIM = 64  # GNN 隐藏层维度
GNN_OUTPUT_DIM = 256  # GNN 输出维度
GNN_NUM_LAYERS = 3  # GNN 层数
GNN_HEADS = 4  # 注意力头数
GNN_DROPOUT = 0.1  # GNN Dropout率

# 世界模型配置
WORLD_MODEL_INPUT_DIM = 256  # 世界模型输入维度
WORLD_MODEL_HIDDEN_DIM = 128  # 世界模型隐藏层维度
WORLD_MODEL_FUTURE_STEPS = 5  # 预测未来步数
WORLD_MODEL_NUM_LAYERS = 2  # 世界模型层数
WORLD_MODEL_DROPOUT = 0.1  # 世界模型 Dropout率

# 控制器配置
CONTROLLER_GNN_DIM = 256  # 控制器GNN维度
CONTROLLER_WORLD_DIM = 256  # 控制器世界模型维度
CONTROLLER_GLOBAL_DIM = 16  # 全局统计特征维度
CONTROLLER_HIDDEN_DIM = 128  # 控制器隐藏层维度
CONTROLLER_ACTION_DIM = 2  # 动作维度（加速度、换道）
CONTROLLER_TOP_K = 5  # Top-K风险车辆数
CONTROLLER_DROPOUT = 0.2  # 控制器Dropout率

# 特征提取器配置
FEATURE_EXTRACTOR_OUTPUT_DIM = 512  # 特征提取器输出维度

# ============================================================================
# 安全约束常量
# ============================================================================

# TTC (Time to Collision) 阈值
DEFAULT_TTC_THRESHOLD = 2.0  # 默认TTC阈值（秒）

# THW (Time Headway) 阈值
DEFAULT_THW_THRESHOLD = 1.5  # 默认THW阈值（秒）

# 加速度限制
DEFAULT_MAX_ACCEL = 2.0  # 最大加速度 (m/s²)
DEFAULT_MAX_DECEL = -3.0  # 最大减速度 (m/s²)
EMERGENCY_DECEL = -5.0  # 紧急制动减速度 (m/s²)

# 空气阻力参数
AIR_DRAG_COEFFICIENT = 0.3  # 空气阻力系数
VEHICLE_MASS = 1500.0  # 车辆质量 (kg)

# ============================================================================
# 训练配置常量
# ============================================================================

# PPO 训练参数
DEFAULT_LEARNING_RATE = 1e-4  # 默认学习率
PPO_GAMMA = 0.99  # 折扣因子
PPO_GAE_LAMBDA = 0.95  # GAE lambda参数
PPO_CLIP_EPSILON = 0.2  # PPO裁剪参数
PPO_ENTROPY_COEF = 0.01  # 熵系数
PPO_VALUE_LOSS_COEF = 0.5  # 价值损失系数
PPO_UPDATE_EPOCHS = 10  # PPO更新轮数
PPO_BATCH_SIZE = 64  # PPO批次大小

# 数据收集参数
DEFAULT_CONTROL_RATIO = 0.25  # ICV（智能网联车）控制比例
DEFAULT_DATA_COLLECTION_TIMEOUT = 180  # 数据收集超时时间（秒）

# ============================================================================
# 评估配置常量
# ============================================================================

DEFAULT_EVAL_EPISODES = 5  # 默认评估episodes数量

# ============================================================================
# 日志和检查点配置常量
# ============================================================================

LOG_INTERVAL = 10  # 日志记录间隔（步数）
SAVE_INTERVAL = 100  # 检查点保存间隔（步数）
SAVE_TOP_K = 3  # 保留最好的K个检查点

# ============================================================================
# 文件路径常量
# ============================================================================

DEFAULT_DATA_DIR = "data"
DEFAULT_LOG_DIR = "logs"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_RESULT_DIR = "results"

# ============================================================================
# 设备配置常量
# ============================================================================

# 设备优先级（按顺序检查可用性）
DEVICE_PREFERENCES = ["cuda", "mps", "cpu"]

# ============================================================================
# 其他常量
# ============================================================================

# 随机种子
DEFAULT_SEED = 42

# 数值稳定性常数
EPSILON = 1e-8  # 小值，防止除零
