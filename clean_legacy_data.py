#!/usr/bin/env python3
"""
清理课程学习遗留数据和临时文件

功能：
1. 清理所有__pycache__目录和.pyc文件
2. 清理检查点文件（如果存在）
3. 清理日志和TensorBoard事件文件（如果存在）
4. 清理SUMO临时文件
5. 清理Python缓存文件

使用方法：
    python clean_legacy_data.py
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 配置路径
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
TENSORBOARD_DIR = PROJECT_ROOT / "runs"


class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text: str):
    """打印成功信息"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """打印信息"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def get_size(path: Path) -> int:
    """获取文件或目录大小（字节）"""
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return 0


def format_size(size_bytes: int) -> str:
    """格式化大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def remove_directory(path: Path, description: str) -> bool:
    """删除目录"""
    if not path.exists():
        print_info(f"{description} 不存在，跳过")
        return False

    try:
        size = get_size(path)
        shutil.rmtree(path)
        print_success(f"已清理 {description}: {path} ({format_size(size)})")
        return True
    except Exception as e:
        print_error(f"清理 {description} 失败: {e}")
        return False


def remove_files(pattern: str, description: str) -> int:
    """删除匹配的文件"""
    files = list(PROJECT_ROOT.rglob(pattern))

    if not files:
        print_info(f"{description} 不存在，跳过")
        return 0

    total_size = 0
    removed_count = 0

    for file in files:
        try:
            size = get_size(file)
            file.unlink()
            total_size += size
            removed_count += 1
        except Exception as e:
            print_warning(f"删除文件失败 {file}: {e}")

    if removed_count > 0:
        print_success(f"已清理 {description}: {removed_count} 个文件 ({format_size(total_size)})")

    return removed_count


def clean_pycache():
    """清理Python缓存"""
    print_header("1. 清理 Python 缓存")

    # 删除所有__pycache__目录
    pycache_dirs = list(PROJECT_ROOT.rglob('__pycache__'))
    removed_count = 0
    total_size = 0

    for dir_path in pycache_dirs:
        try:
            size = get_size(dir_path)
            shutil.rmtree(dir_path)
            total_size += size
            removed_count += 1
        except Exception as e:
            print_warning(f"删除 __pycache__ 失败 {dir_path}: {e}")

    if removed_count > 0:
        print_success(f"已清理 __pycache__ 目录: {removed_count} 个 ({format_size(total_size)})")
    else:
        print_info("没有找到 __pycache__ 目录")

    # 删除.pyc文件
    remove_files("*.pyc", "Python .pyc 文件")
    remove_files("*.pyo", "Python .pyo 文件")


def clean_checkpoints():
    """清理检查点文件"""
    print_header("2. 清理检查点文件")

    # 清理整个checkpoints目录
    remove_directory(CHECKPOINT_DIR, "检查点目录")

    # 清理其他可能的检查点文件
    checkpoint_files = list(PROJECT_ROOT.rglob("*.zip")) + list(PROJECT_ROOT.rglob("*.ckpt"))
    removed_count = 0
    total_size = 0

    for file in checkpoint_files:
        try:
            size = get_size(file)
            file.unlink()
            total_size += size
            removed_count += 1
            print_success(f"已删除检查点: {file.relative_to(PROJECT_ROOT)} ({format_size(size)})")
        except Exception as e:
            print_warning(f"删除检查点失败 {file}: {e}")

    if removed_count > 0:
        print_success(f"共清理 {removed_count} 个检查点文件 ({format_size(total_size)})")
    else:
        print_info("没有找到检查点文件")


def clean_logs():
    """清理日志文件"""
    print_header("3. 清理日志文件")

    # 清理logs目录
    remove_directory(LOG_DIR, "日志目录")

    # 清理TensorBoard目录
    remove_directory(TENSORBOARD_DIR, "TensorBoard目录")

    # 清理.log文件
    log_files = list(PROJECT_ROOT.rglob("*.log"))
    removed_count = 0
    total_size = 0

    for file in log_files:
        try:
            size = get_size(file)
            file.unlink()
            total_size += size
            removed_count += 1
        except Exception as e:
            print_warning(f"删除日志失败 {file}: {e}")

    if removed_count > 0:
        print_success(f"已清理 .log 文件: {removed_count} 个 ({format_size(total_size)})")
    else:
        print_info("没有找到 .log 文件")


def clean_sumo_temp():
    """清理SUMO临时文件"""
    print_header("4. 清理 SUMO 临时文件")

    # SUMO临时文件模式
    temp_patterns = [
        "*.xml.bak",
        "*.xml.old",
        "*.net.cc",
        "*.net.xml.gz",
        "vclass_*",
        "*.emit.xml",
        "*.stat.xml",
    ]

    total_removed = 0
    total_size = 0

    for pattern in temp_patterns:
        files = list(PROJECT_ROOT.rglob(pattern))
        for file in files:
            try:
                size = get_size(file)
                file.unlink()
                total_size += size
                total_removed += 1
            except Exception as e:
                print_warning(f"删除SUMO临时文件失败 {file}: {e}")

    if total_removed > 0:
        print_success(f"已清理 SUMO 临时文件: {total_removed} 个 ({format_size(total_size)})")
    else:
        print_info("没有找到 SUMO 临时文件")


def clean_cache_files():
    """清理其他缓存文件"""
    print_header("5. 清理其他缓存文件")

    # 清理.pytest_cache
    remove_directory(PROJECT_ROOT / ".pytest_cache", "pytest缓存")

    # 清理.hypothesis
    remove_directory(PROJECT_ROOT / ".hypothesis", "hypothesis缓存")

    # 清理.mypy_cache
    remove_directory(PROJECT_ROOT / ".mypy_cache", "mypy缓存")

    # 清理.DS_Store (macOS)
    ds_store_files = list(PROJECT_ROOT.rglob(".DS_Store"))
    for file in ds_store_files:
        try:
            file.unlink()
            print_success(f"已删除 .DS_Store: {file.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print_warning(f"删除 .DS_Store 失败 {file}: {e}")


def show_summary():
    """显示清理总结"""
    print_header("清理完成！")

    # 计算剩余的重要目录大小
    important_dirs = {
        "src": PROJECT_ROOT / "src",
        "configs": PROJECT_ROOT / "configs",
        "仿真环境": PROJECT_ROOT / "仿真环境_初赛_1.0",
    }

    print(f"\n{Colors.OKCYAN}重要目录大小:{Colors.ENDC}\n")
    for name, path in important_dirs.items():
        if path.exists():
            size = get_size(path)
            print(f"  {name:20s}: {format_size(size)}")

    print(f"\n{Colors.OKGREEN}✓ 所有遗留数据和临时文件已清理完毕！{Colors.ENDC}\n")
    print(f"{Colors.OKCYAN}建议: 现在可以开始全新的训练了{Colors.ENDC}\n")


def main():
    """主函数"""
    print_header("清理课程学习遗留数据")

    print_info("项目目录: " + str(PROJECT_ROOT))
    print_warning("此操作将删除所有缓存、检查点和日志文件")

    # 确认
    response = input("\n是否继续? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print_info("已取消清理")
        return

    # 执行清理
    clean_pycache()
    clean_checkpoints()
    clean_logs()
    clean_sumo_temp()
    clean_cache_files()

    # 显示总结
    show_summary()


if __name__ == "__main__":
    main()
