#!/usr/bin/env python3
"""
分析项目中的静默错误处理
"""
import re
from pathlib import Path

# 需要检查的目录
directories = ['src/env', 'src/mpc', 'scripts']

# 忽略的文件
ignore_patterns = ['.pyc', '__pycache__', '.git']

def find_silent_exceptions():
    """查找所有静默错误处理"""
    results = []

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob('*.py'):
            # 跳过一些不重要的文件
            if any(pattern in str(py_file) for pattern in ignore_patterns):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    # 检测 except: 后面跟 pass/continue/return
                    if re.search(r'except:\s*$', line):
                        # 检查下一行是否是 pass/continue/return
                        if i < len(lines):
                            next_line = lines[i].strip()
                            if next_line in ['pass', 'continue'] or next_line.startswith('return '):
                                # 获取上下文
                                context_start = max(0, i - 3)
                                context_end = min(len(lines), i + 2)
                                context = ''.join(lines[context_start:context_end])

                                results.append({
                                    'file': str(py_file),
                                    'line': i,
                                    'code': line.strip(),
                                    'next_line': next_line,
                                    'context': context.strip()
                                })
            except Exception as e:
                print(f"Error reading {py_file}: {e}")

    return results

def categorize_severity(result):
    """分类严重程度"""
    file = result['file']
    code = result['code']

    # 关键路径：数据收集、环境交互、控制逻辑
    critical_paths = ['collect', 'env', 'mpc', 'control']
    # 可忽略路径：清理、关闭、初始化
    ignore_paths = ['close', 'cleanup', 'init', 'import']

    file_lower = file.lower()

    # 严重：关键路径中的静默错误
    if any(path in file_lower for path in critical_paths):
        # 除了close/cleanup/import
        if not any(path in file_lower for path in ignore_paths):
            return 'HIGH'

    # 中等：其他位置
    return 'MEDIUM'

def main():
    print("=== 静默错误处理分析 ===\n")

    results = find_silent_exceptions()

    if not results:
        print("✓ 没有发现静默错误处理")
        return

    # 按严重程度分类
    high_severity = []
    medium_severity = []

    for result in results:
        severity = categorize_severity(result)
        if severity == 'HIGH':
            high_severity.append(result)
        else:
            medium_severity.append(result)

    print(f"总计: {len(results)} 个静默错误处理")
    print(f"  🔴 高风险: {len(high_severity)} 个")
    print(f"  🟡 中风险: {len(medium_severity)} 个")
    print()

    if high_severity:
        print("=" * 80)
        print("🔴 高风险静默错误处理（关键路径）")
        print("=" * 80)
        for result in high_severity:
            print(f"\n文件: {result['file']}:{result['line']}")
            print(f"代码: {result['code']}")
            print(f"上下文:\n{result['context']}\n")

    if medium_severity:
        print("\n" + "=" * 80)
        print("🟡 中风险静默错误处理（非关键路径）")
        print("=" * 80)
        for result in medium_severity[:10]:  # 只显示前10个
            print(f"\n文件: {result['file']}:{result['line']}")
            print(f"代码: {result['code']}")

    # 建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    print("""
1. 高风险静默错误需要修复：
   - 添加日志输出（至少使用logger.warning）
   - 或者抛出异常（如果这是关键错误）

2. 中风险静默错误可以保留：
   - 如果错误是预期的且可恢复的
   - 添加注释说明为什么忽略

3. 示例修复：
   修复前: except: pass
   修复后: except Exception as e: logger.warning(f"Failed to get vehicle state: {e}")

   修复前: except: return 0.0
   修复后: except Exception as e: logger.debug(f"Edge lookup failed: {e}"); return 0.0
""")

if __name__ == '__main__':
    main()
