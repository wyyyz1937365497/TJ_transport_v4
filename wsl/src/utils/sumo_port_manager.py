"""
SUMO端口管理工具
解决端口冲突问题
"""

import socket
import subprocess
import time
from typing import Optional, List


def find_free_port(start_port: int = 8813, max_attempts: int = 100) -> int:
    """
    查找可用端口

    Args:
        start_port: 起始端口
        max_attempts: 最大尝试次数

    Returns:
        可用端口号
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"无法找到可用端口 (尝试了 {max_attempts} 次)")


def kill_sumo_processes():
    """杀死所有SUMO进程"""
    try:
        # WSL/Linux
        subprocess.run(['killall', '-9', 'sumo'], stderr=subprocess.DEVNULL)
        subprocess.run(['killall', '-9', 'sumo-gui'], stderr=subprocess.DEVNULL)
        # Windows (通过WSL)
        subprocess.run(['killall', '-9', 'sumo.exe'], stderr=subprocess.DEVNULL)
    except:
        pass


def check_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return False
    except OSError:
        return True


def cleanup_ports():
    """清理被占用的SUMO端口"""
    print("正在清理SUMO端口...")
    kill_sumo_processes()

    # 等待端口释放
    for port in range(8813, 8850):
        if check_port_in_use(port):
            print(f"  端口 {port} 仍被占用，等待释放...")
            for _ in range(10):
                time.sleep(0.5)
                if not check_port_in_use(port):
                    break

    print("端口清理完成")


class PortManager:
    """端口管理器 - 为每个SUMO实例分配唯一端口"""

    def __init__(self, start_port: int = 8813):
        self.start_port = start_port
        self.used_ports = set()
        self.lock = threading.Lock()

    def acquire_port(self) -> int:
        """获取一个可用端口"""
        with self.lock:
            port = self.start_port + len(self.used_ports)
            # 确保端口可用
            while check_port_in_use(port):
                port += 1
            self.used_ports.add(port)
            return port

    def release_port(self, port: int):
        """释放端口"""
        with self.lock:
            self.used_ports.discard(port)


import threading
