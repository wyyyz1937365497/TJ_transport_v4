#!/usr/bin/env python3
"""
从ss.txt中提取每一次迭代的训练输出信息
"""

import re
import json
from pathlib import Path


def extract_iterations(log_file_path):
    """
    提取日志文件中每一次迭代的关键信息
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        包含每次迭代信息的列表
    """
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则表达式模式，用于匹配迭代块
    # 匹配 "迭代 X/Y" 到下一个迭代或文件末尾
    iteration_pattern = r'迭代\s+(\d+)/(\d+)'
    
    iterations = []
    
    # 找到所有迭代的开始位置
    iteration_starts = []
    for match in re.finditer(iteration_pattern, content):
        iteration_num = int(match.group(1))
        total_iters = int(match.group(2))
        start_pos = match.start()
        iteration_starts.append({
            'num': iteration_num,
            'total': total_iters,
            'pos': start_pos
        })
    
    # 为每个迭代提取信息
    for i, iter_info in enumerate(iteration_starts):
        iter_num = iter_info['num']
        start_pos = iter_info['pos']
        
        # 确定迭代块的结束位置
        if i < len(iteration_starts) - 1:
            end_pos = iteration_starts[i + 1]['pos']
        else:
            end_pos = len(content)
        
        # 提取迭代块内容
        iter_block = content[start_pos:end_pos]
        
        # 提取关键指标
        metrics = {}
        
        # 提取 Policy Loss
        policy_loss_match = re.search(r'Policy Loss:\s+([\d.]+)', iter_block)
        if policy_loss_match:
            metrics['Policy Loss'] = float(policy_loss_match.group(1))
        
        # 提取 Value Loss
        value_loss_match = re.search(r'Value Loss:\s+([\d.]+)', iter_block)
        if value_loss_match:
            metrics['Value Loss'] = float(value_loss_match.group(1))
        
        # 提取 Entropy
        entropy_match = re.search(r'Entropy:\s+([\d.]+)', iter_block)
        if entropy_match:
            metrics['Entropy'] = float(entropy_match.group(1))
        
        # 提取 Mean Reward
        mean_reward_match = re.search(r'Mean Reward:\s+([\d.-]+)', iter_block)
        if mean_reward_match:
            metrics['Mean Reward'] = float(mean_reward_match.group(1))
        
        # 提取 Mean Return
        mean_return_match = re.search(r'Mean Return:\s+([\d.-]+)', iter_block)
        if mean_return_match:
            metrics['Mean Return'] = float(mean_return_match.group(1))
        
        # 提取 LR (Learning Rate)
        lr_match = re.search(r'LR:\s+([\d.e-]+)', iter_block)
        if lr_match:
            metrics['LR'] = float(lr_match.group(1))
        
        iteration_data = {
            'iteration': iter_num,
            'metrics': metrics
        }
        
        iterations.append(iteration_data)
    
    return iterations


def main():
    """主函数"""
    log_file = Path('/home/wyyyz/TJ_transport_v4/logs/ss.txt')
    
    if not log_file.exists():
        print(f"错误: 文件不存在 {log_file}")
        return
    
    print("正在提取迭代信息...")
    iterations = extract_iterations(log_file)
    
    # 输出为 JSON 格式
    output_file = log_file.parent / 'iterations_metrics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(iterations, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已提取 {len(iterations)} 次迭代的信息")
    print(f"✓ 结果已保存到: {output_file}")
    
    # 打印前几个迭代的摘要
    print("\n前5次迭代的摘要:")
    print("-" * 80)
    for iter_data in iterations[:5]:
        print(f"迭代 {iter_data['iteration']}:")
        for key, value in iter_data['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print()
    


if __name__ == '__main__':
    main()
