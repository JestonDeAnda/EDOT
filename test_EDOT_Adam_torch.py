#!/usr/bin/env python
'''
test_EDOT_Adam_torch.py

测试PyTorch版本的EDOT Adam优化器实现
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.EDOT_Adam_torch import EDOT_Torch
from src.edot_ET_adam import EDOT as EDOT_Numpy

# 确保结果目录存在
os.makedirs("test_results", exist_ok=True)


def test_basic_functionality():
    """
    测试基本功能：确保EDOT_Torch类可以正确初始化和运行
    """
    print("\n=== 测试基本功能 ===")
    edot_torch = EDOT_Torch()
    assert edot_torch is not None, "EDOT_Torch初始化失败"
    print("✅ EDOT_Torch初始化成功")


def test_adam_discretize_small():
    """
    在小数据集上测试adam_discretize函数
    """
    print("\n=== 测试adam_discretize（小数据集）===")

    # 创建一个小的测试数据集
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成随机点
    n_points = 100
    points = np.random.rand(n_points, 2)
    weights = np.ones(n_points) / n_points

    # 初始离散点
    discrete_size = 3
    disc = np.random.rand(discrete_size, 2)

    # 转换为PyTorch张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_torch = torch.tensor(points, dtype=torch.float32, device=device)
    weights_torch = torch.tensor(weights, dtype=torch.float32, device=device)
    disc_torch = torch.tensor(disc, dtype=torch.float32, device=device)

    # 运行PyTorch版本
    edot_torch = EDOT_Torch()

    # 测试两种模式
    for use_pytorch_adam in [True, False]:
        print(f"\n使用PyTorch Adam: {use_pytorch_adam}")
        edot_torch.wrap_grad(edot_torch._grad,
                             [50 + x + i for i in range(50) for x in range(2)],
                             0.1)
        start_time = time.time()
        X_torch, W_torch, loss_torch, L_torch = edot_torch.adam_discretize(
            disc_torch,
            points_torch,
            weights_torch,
            repeats=100,  # 减少迭代次数以加快测试
            alpha=0.01,
            zeta=0.01,
            use_pytorch_adam=use_pytorch_adam)
        torch_time = time.time() - start_time

        print(f"PyTorch版本运行时间: {torch_time:.4f}秒")
        print(f"最终损失: {loss_torch:.6f}, 梯度范数: {L_torch:.6f}")
        print(f"离散点坐标:\n{X_torch}")
        print(f"离散点权重:\n{W_torch}")

        # 验证权重和为1
        assert abs(W_torch.sum().item() - 1.0) < 1e-6, "权重和不为1"
        print("✅ 权重和为1")

        # 验证坐标在[0,1]范围内
        assert torch.all((X_torch >= 0) & (X_torch <= 1)), "坐标不在[0,1]范围内"
        print("✅ 坐标在[0,1]范围内")


def test_discretize_from_coordinates():
    """
    测试discretize_from_coordinates函数
    """
    print("\n=== 测试discretize_from_coordinates ===")

    # 创建一个小的测试数据集
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成随机点
    n_points = 100
    points = np.random.rand(n_points, 2) * 1000  # 模拟像素坐标

    # 运行PyTorch版本
    edot_torch = EDOT_Torch()

    # 测试两种模式
    for use_pytorch_adam in [True, False]:
        print(f"\n使用PyTorch Adam: {use_pytorch_adam}")

        start_time = time.time()
        centers, weights, labels, loss, L = edot_torch.discretize_from_coordinates(
            points,
            discrete_size=3,
            repeats=100,  # 减少迭代次数以加快测试
            alpha=0.01,
            zeta=0.01,
            use_pytorch_adam=use_pytorch_adam)
        torch_time = time.time() - start_time

        print(f"PyTorch版本运行时间: {torch_time:.4f}秒")
        print(f"最终损失: {loss:.6f}, 梯度范数: {L:.6f}")
        print(f"中心点数量: {len(centers)}")
        print(f"标签数量: {len(labels)}")

        # 验证标签数量与点数量相同
        assert len(labels) == n_points, "标签数量与点数量不同"
        print("✅ 标签数量与点数量相同")

        # 验证权重和为1
        assert abs(np.sum(weights) - 1.0) < 1e-6, "权重和不为1"
        print("✅ 权重和为1")

        # 可视化结果
        plt.figure(figsize=(10, 8))
        plt.scatter(points[:, 0],
                    points[:, 1],
                    c=labels,
                    alpha=0.7,
                    cmap='tab10')
        plt.scatter(centers[:, 0],
                    centers[:, 1],
                    c='red',
                    s=weights * 1000,
                    marker='X')

        for i, center in enumerate(centers):
            plt.text(center[0],
                     center[1],
                     f'{i+1}',
                     fontsize=12,
                     ha='center',
                     va='center',
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.title(f"EDOT Discretization (PyTorch Adam: {use_pytorch_adam})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)

        save_path = f"test_results/discretize_pytorch_adam_{use_pytorch_adam}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 结果已保存至 {save_path}")


def test_compare_with_numpy():
    """
    比较PyTorch版本和NumPy版本的结果
    """
    print("\n=== 比较PyTorch版本和NumPy版本 ===")

    # 创建一个小的测试数据集
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成随机点
    n_points = 100
    points = np.random.rand(n_points, 2) * 1000  # 模拟像素坐标

    # 初始离散点 - 确保两个版本使用相同的初始值
    discrete_size = 3
    initial_disc = np.random.rand(discrete_size, 2) * 0.9 + 0.05

    # 运行NumPy版本
    print("\n运行NumPy版本...")
    edot_numpy = EDOT_Numpy()
    start_time = time.time()
    centers_numpy, weights_numpy, labels_numpy, loss_numpy, L_numpy = edot_numpy.discretize_from_coordinates(
        points,
        disc=initial_disc,
        discrete_size=discrete_size,
        repeats=100,  # 减少迭代次数以加快测试
        alpha=0.01,
        zeta=0.01)
    numpy_time = time.time() - start_time
    print(f"NumPy版本运行时间: {numpy_time:.4f}秒")

    # 运行PyTorch版本
    print("\n运行PyTorch版本...")
    edot_torch = EDOT_Torch()
    start_time = time.time()
    centers_torch, weights_torch, labels_torch, loss_torch, L_torch = edot_torch.discretize_from_coordinates(
        points,
        disc=initial_disc,
        discrete_size=discrete_size,
        repeats=100,  # 减少迭代次数以加快测试
        alpha=0.01,
        zeta=0.01,
        use_pytorch_adam=True)
    torch_time = time.time() - start_time
    print(f"PyTorch版本运行时间: {torch_time:.4f}秒")

    # 比较结果
    print("\n比较结果:")
    print(f"NumPy损失: {loss_numpy:.6f}, PyTorch损失: {loss_torch:.6f}")
    print(f"NumPy梯度范数: {L_numpy:.6f}, PyTorch梯度范数: {L_torch:.6f}")

    # 可视化比较
    plt.figure(figsize=(15, 7))

    # NumPy结果
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0],
                points[:, 1],
                c=labels_numpy,
                alpha=0.7,
                cmap='tab10')
    plt.scatter(centers_numpy[:, 0],
                centers_numpy[:, 1],
                c='red',
                s=weights_numpy * 1000,
                marker='X')
    for i, center in enumerate(centers_numpy):
        plt.text(center[0],
                 center[1],
                 f'{i+1}',
                 fontsize=12,
                 ha='center',
                 va='center',
                 bbox=dict(facecolor='white', alpha=0.8))
    plt.title(f"NumPy版本 (Loss: {loss_numpy:.4f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)

    # PyTorch结果
    plt.subplot(1, 2, 2)
    plt.scatter(points[:, 0],
                points[:, 1],
                c=labels_torch,
                alpha=0.7,
                cmap='tab10')
    plt.scatter(centers_torch[:, 0],
                centers_torch[:, 1],
                c='red',
                s=weights_torch * 1000,
                marker='X')
    for i, center in enumerate(centers_torch):
        plt.text(center[0],
                 center[1],
                 f'{i+1}',
                 fontsize=12,
                 ha='center',
                 va='center',
                 bbox=dict(facecolor='white', alpha=0.8))
    plt.title(f"PyTorch版本 (Loss: {loss_torch:.4f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "test_results/numpy_vs_pytorch_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 比较结果已保存至 {save_path}")

    # 计算加速比
    speedup = numpy_time / torch_time
    print(f"PyTorch版本比NumPy版本快 {speedup:.2f} 倍")


def test_gpu_support():
    """
    测试GPU支持
    """
    print("\n=== 测试GPU支持 ===")

    if not torch.cuda.is_available():
        print("⚠️ 没有可用的GPU，跳过GPU测试")
        return

    print(f"发现GPU: {torch.cuda.get_device_name(0)}")

    # 创建一个小的测试数据集
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成随机点
    n_points = 1000  # 使用更多点以更好地测试GPU性能
    points = np.random.rand(n_points, 2) * 1000

    # 运行CPU版本
    print("\n在CPU上运行...")
    with torch.device('cpu'):
        edot_cpu = EDOT_Torch()
        start_time = time.time()
        _, _, _, loss_cpu, _ = edot_cpu.discretize_from_coordinates(
            points,
            discrete_size=5,
            repeats=50,
            alpha=0.01,
            zeta=0.01,
            use_pytorch_adam=True)
        cpu_time = time.time() - start_time
    print(f"CPU运行时间: {cpu_time:.4f}秒")

    # 运行GPU版本
    print("\n在GPU上运行...")
    edot_gpu = EDOT_Torch()
    start_time = time.time()
    _, _, _, loss_gpu, _ = edot_gpu.discretize_from_coordinates(
        points,
        discrete_size=5,
        repeats=50,
        alpha=0.01,
        zeta=0.01,
        use_pytorch_adam=True)
    gpu_time = time.time() - start_time
    print(f"GPU运行时间: {gpu_time:.4f}秒")

    # 比较结果
    speedup = cpu_time / gpu_time
    print(f"GPU比CPU快 {speedup:.2f} 倍")
    print(f"CPU损失: {loss_cpu:.6f}, GPU损失: {loss_gpu:.6f}")


if __name__ == "__main__":
    print("开始测试EDOT_Adam_torch...")

    # 运行测试
    # test_basic_functionality()
    test_adam_discretize_small()
    # test_discretize_from_coordinates()
    # test_compare_with_numpy()
    # test_gpu_support()

    print("\n所有测试完成!")
