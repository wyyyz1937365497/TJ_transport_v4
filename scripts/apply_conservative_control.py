#!/usr/bin/env python3
"""
保守控制补丁 - 快速修复OCR低于baseline的问题

用法：
    python scripts/apply_conservative_control.py

这个脚本会：
1. 备份原始submit_solution.py
2. 添加保守控制过滤（禁用减速和换道）
3. 创建可测试的修改版本
"""

import shutil
from pathlib import Path


def apply_conservative_control_patch():
    """应用保守控制补丁"""

    print("=" * 70)
    print("保守控制补丁应用工具")
    print("=" * 70)

    # 文件路径
    submit_file = Path("submit_solution.py")
    backup_file = Path("submit_solution.py.backup_before_conservative")

    # 备份原始文件
    if not backup_file.exists():
        print(f"\n[步骤1] 备份原始文件...")
        shutil.copy(submit_file, backup_file)
        print(f"✓ 已备份到: {backup_file}")
    else:
        print(f"\n[步骤1] 备份已存在: {backup_file}")

    # 读取原始文件
    print(f"\n[步骤2] 读取原始文件...")
    with open(submit_file, 'r') as f:
        content = f.read()

    # 检查是否已经应用过补丁
    if "filter_conservative_actions" in content:
        print("⚠️  补丁已应用，无需重复应用")
        return

    # 找到apply_control_algorithm方法
    print(f"\n[步骤3] 应用保守控制补丁...")

    # 添加保守控制方法（在OCRMAXSubmission类中）
    method_to_add = '''

    def filter_conservative_actions(self, actions_dict, vehicle_states):
        """
        保守控制过滤：只保留安全的控制动作

        策略：
        1. 禁止减速（只允许加速或保持）
        2. 禁止换道（避免干扰交通流）

        目标：减少有害控制，使OCR恢复到baseline水平或更高
        """
        safe_actions = {}

        for veh_id, action in actions_dict.items():
            acceleration, lane_change = action

            # 1. 禁止减速（只允许加速或保持）
            if acceleration < 0:
                acceleration = 0.0

            # 2. 禁止换道（设为0，不执行换道）
            lane_change = 0.0

            safe_actions[veh_id] = (acceleration, lane_change)

        return safe_actions
'''

    # 在apply_control_algorithm方法中添加过滤调用
    # 找到应用控制动作的位置
    target_line = "self.apply_control_actions(actions_dict)"

    if target_line not in content:
        print("❌ 未找到目标行，可能submit_solution.py已被修改")
        return

    # 替换目标行，添加保守控制过滤
    replacement = '''# ========== 保守控制过滤（禁用减速和换道）==========
            actions_dict = self.filter_conservative_actions(actions_dict, obs_dict)

            self.apply_control_actions(actions_dict)'''

    content = content.replace(target_line, replacement)

    # 在类中添加新方法（找到apply_control_algorithm方法的定义之后）
    # 找到apply_control_algorithm方法的位置
    method_def = "    def apply_control_algorithm(self, step):"
    method_end_pos = content.find("\n    def ", content.find(method_def) + 100)

    if method_end_pos == -1:
        print("❌ 未找到方法插入位置")
        return

    # 在apply_control_algorithm方法之后插入新方法
    content = content[:method_end_pos] + method_to_add + content[method_end_pos:]

    # 写入修改后的文件
    print(f"\n[步骤4] 写入修改后的文件...")
    with open(submit_file, 'w') as f:
        f.write(content)

    print(f"✓ 补丁已应用")

    print("\n" + "=" * 70)
    print("补丁应用成功！")
    print("=" * 70)

    print("\n📋 修改内容：")
    print("  1. 添加了filter_conservative_actions方法")
    print("     - 禁止减速（acceleration < 0 → 0）")
    print("     - 禁止换道（lane_change → 0）")
    print("  2. 在apply_control_algorithm中调用过滤方法")

    print("\n🧪 测试步骤：")
    print("  1. 运行提交脚本：python submit_solution.py")
    print("  2. 查看OCR是否提升：")
    print("     python scripts/calculate_competition_score.py --from_baseline_file")
    print("  3. 如果OCR > 54.69%，可以提交！")
    print("  4. 如果OCR < 54.69%，尝试其他方案（见EXECUTIVE_SUMMARY.md）")

    print("\n📁 备份文件：")
    print(f"  {backup_file}")
    print("  如果需要回退，运行：")
    print(f"    cp {backup_file} {submit_file}")

    print("\n" + "=" * 70 + "\n")


def apply_lower_penetration_patch():
    """应用降低渗透率补丁"""

    print("=" * 70)
    print("降低渗透率补丁应用工具")
    print("=" * 70)

    # 文件路径
    submit_file = Path("submit_solution.py")
    backup_file = Path("submit_solution.py.backup_before_lower_penetration")

    # 备份原始文件
    if not backup_file.exists():
        print(f"\n[步骤1] 备份原始文件...")
        shutil.copy(submit_file, backup_file)
        print(f"✓ 已备份到: {backup_file}")
    else:
        print(f"\n[步骤1] 备份已存在: {backup_file}")

    # 读取原始文件
    print(f"\n[步骤2] 读取原始文件...")
    with open(submit_file, 'r') as f:
        content = f.read()

    # 检查是否已经应用过补丁
    if "icv_ratio = 0.05" in content or "icv_ratio = 0.08" in content:
        print("⚠️  补丁已应用，无需重复应用")
        return

    print(f"\n[步骤3] 应用降低渗透率补丁...")

    # 替换渗透率
    # 10% → 5%
    if "icv_ratio = 0.10" in content:
        content = content.replace("icv_ratio = 0.10", "icv_ratio = 0.05")
        new_ratio = "5%"
    elif "icv_ratio = 0.1" in content:
        content = content.replace("icv_ratio = 0.1", "icv_ratio = 0.05")
        new_ratio = "5%"
    else:
        print("❌ 未找到icv_ratio配置行")
        return

    # 写入修改后的文件
    print(f"\n[步骤4] 写入修改后的文件...")
    with open(submit_file, 'w') as f:
        f.write(content)

    print(f"✓ 补丁已应用")

    print("\n" + "=" * 70)
    print("补丁应用成功！")
    print("=" * 70)

    print(f"\n📋 修改内容：")
    print(f"  ICV渗透率：10% → {new_ratio}")
    print(f"  控制车辆数：~25 → ~12-13")

    print("\n🧪 测试步骤：")
    print("  1. 运行提交脚本：python submit_solution.py")
    print("  2. 查看OCR是否提升")

    print("\n📁 备份文件：")
    print(f"  {backup_file}")

    print("\n" + "=" * 70 + "\n")


def main():
    """主函数"""
    import sys

    print("\n🔧 OCR-MAX快速修复工具\n")

    if len(sys.argv) > 1:
        patch_type = sys.argv[1]

        if patch_type == "conservative":
            apply_conservative_control_patch()
        elif patch_type == "lower_penetration":
            apply_lower_penetration_patch()
        else:
            print("❌ 未知的补丁类型")
            print("\n用法：")
            print("  python scripts/apply_patches.py conservative      # 保守控制")
            print("  python scripts/apply_patches.py lower_penetration # 降低渗透率")
    else:
        print("请选择要应用的补丁：")
        print("\n1. 保守控制补丁（推荐，优先尝试）")
        print("   - 禁止减速")
        print("   - 禁止换道")
        print("   - 预期OCR提升到54.5-54.8%")
        print("\n2. 降低渗透率补丁")
        print("   - ICV比例：10% → 5%")
        print("   - 控制车辆：~25 → ~12-13")
        print("   - 预期OCR提升到54.5-54.9%")
        print("\n用法：")
        print("  python scripts/apply_patches.py conservative")
        print("  python scripts/apply_patches.py lower_penetration")


if __name__ == '__main__':
    main()
