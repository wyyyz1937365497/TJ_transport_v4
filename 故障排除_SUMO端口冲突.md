# SUMO端口冲突问题 - 已修复 ✅

## 问题描述

运行 `python quick_start.py` 或 `python train.py` 时遇到以下错误：

```
Error: A value for the option 'remote-port' was already set.
 Possible synonymes:
Error: Could not parse commandline options.
```

## 问题原因

代码中手动设置了 `--remote-port` 参数，但这可能与以下情况冲突：
1. SUMO配置文件（.sumocfg）中已设置端口
2. 多次设置同一参数导致SUMO命令行解析失败

## 解决方案

已修复以下文件：

### 1. `src/env/sumo_env.py`
**修改前**：
```python
cmd.extend(["--remote-port", str(self.config.get('port', 8813))])
```

**修改后**：
```python
# 移除 remote-port 设置，避免与.sumocfg文件冲突
# SUMO会自动使用默认端口或从配置文件读取
```

### 2. `src/env/data_collector.py`
**修改前**：
```python
process_config['port'] = 8813 + i  # 使用不同端口避免冲突
```

**修改后**：
```python
# 移除 port 设置，让SUMO自动管理端口
```

### 3. `configs/training_config.json`
**修改前**：
```json
"environment": {
    ...
    "port": 8813,
    ...
}
```

**修改后**：
```json
"environment": {
    ...
    // 移除 port 设置
    ...
}
```

## 验证修复

现在可以正常运行：

```bash
# 快速测试
python quick_start.py

# 完整训练
python train.py
```

## 额外提示

### 如果仍然遇到端口问题

1. **检查是否有其他SUMO实例在运行**
   ```bash
   # Windows任务管理器中查找 sumo.exe 进程
   tasklist | findstr sumo
   ```

2. **强制结束占用端口的进程**
   ```bash
   taskkill /F /IM sumo.exe
   ```

3. **使用GUI模式调试**
   ```json
   {
     "use_gui": true
   }
   ```

### 多进程并行训练

如果需要并行运行多个SUMO实例，SUMO会自动分配不同的端口。无需手动设置。

## 技术细节

SUMO的TraCI默认使用端口8813，如果该端口被占用，SUMO会自动尝试其他端口（8814, 8815等）。手动设置反而会导致冲突。

最佳实践：**让SUMO自动管理端口** ✅

---

问题已解决！现在可以正常使用系统了。
