#!/usr/bin/env python
'''
EDOT_Adam_torch

PyTorch implementation of EDOT Adam optimizer.
Requires `torch` and `EDOT_disc_torch`.

Date: 2025.09.05

Python version >= 3.10
'''

import json
import os
from typing import Callable, List
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import EDOT_disc_torch
from .EDOT_disc_torch import gradient_EOT

# 确保结果目录存在
os.makedirs("edot_results_adam_torch", exist_ok=True)


def EDOT_grad_wrapper(
    grad: Callable,
    subsample_sizes: int | List[int] | List[Callable] | None = None,
    zeta: float | list[float] | List[Callable] | None = None,
):
    """
    包装梯度计算函数，支持子采样和正则化。

    参数:
        grad (Callable): 原始的梯度计算函数。
        subsample_sizes (int | List[int] | List[Callable], 可选): 子采样大小。
                    Callable[torch.Tensor] -> torch.Tensor | List, the mask
                    using which can generate the subsample tensor.
        zeta (float | list[float] | List[Callable], 可选): 正则化参数。

    返回:
        Callable: 包装后的梯度计算函数。
    """
    count = 0
    # [TODO] see the other [TODO] below
    zeta = zeta[0] if isinstance(zeta, list) else zeta
    match subsample_sizes:
        case int() if subsample_sizes >= 2:
            mask_gen = lambda x: list(range(min(subsample_sizes, len(x))))

        # list case, 可以是int和callable的混合
        case list() if len(subsample_sizes) > 0:
            count = min(len(subsample_sizes) - 1, count)
            if isinstance(subsample_sizes[count], int):
                mask_gen = lambda x: list(
                    range(min(max(subsample_sizes[count], 2), len(x))))
            elif isinstance(subsample_sizes[count], Callable):
                mask_gen = lambda x: subsample_sizes[count](x)

        case callable():
            mask_gen = lambda x: list(
                range(min(max(subsample_sizes[count], 2), len(x))))
        case _:
            mask_gen = lambda x: list(range(min(subsample_sizes, len(x))))

    def inner(sample, sample_weight, target, target_weight, **kwargs):
        nonlocal count, mask_gen, zeta
        print("Round:", count, f"zeta = {zeta:.5f}")
        _kwargs = {
            k: kwargs[k]
            for k in {"k", "exp", "distance", "distance_gradient"}  # "zeta"
            if k in kwargs
        }
        mask = mask_gen(sample)
        sample = sample[mask]
        ratio = sample_weight.sum() / torch.clamp(sample_weight[mask].sum(),
                                                  min=1e-10)
        sample_weight = sample_weight[mask] * ratio
        # [TODO] 这里的zeta用一个更好的机制来代替
        while True:
            try:
                Dnu, Dy, cost = grad(sample,
                                     sample_weight,
                                     target,
                                     target_weight,
                                     zeta=zeta,
                                     **_kwargs)
            except Exception as e:
                # print(e)
                zeta *= 1.2
                continue
            if not torch.isnan(cost):
                zeta *= 0.8
                break
            zeta *= 1.2

        count += 1
        return Dnu, Dy, cost

    return inner


class EDOT_Torch:

    def __init__(self):
        pass
        self._grad = gradient_EOT
        self.grad = gradient_EOT

    def wrap_grad(self,
                  grad,
                  subsample_sizes,
                  zeta,
                  wrapper=EDOT_grad_wrapper):
        print("Grad Wrapped!")
        self.grad = wrapper(grad, subsample_sizes, zeta)

    def adam_discretize(
        self,
        disc: torch.Tensor,
        positions: torch.Tensor,
        weights: torch.Tensor,
        repeats: int = 5000,
        alpha: float = 0.005,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        zeta: float = 0.004,
        use_pytorch_adam: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        使用Adam优化器离散化EDOT模型
        
        Parameters
        ----------
        disc: torch.Tensor, 初始离散点坐标，形状为(discrete_size, 2)
        positions: torch.Tensor, 样本点坐标，形状为(sample_size, 2)
        weights: torch.Tensor, 样本点权重，形状为(sample_size)
        repeats: int, 最大迭代次数
        alpha: float, 学习率
        beta1: float, 一阶矩估计的指数衰减率
        beta2: float, 二阶矩估计的指数衰减率
        epsilon: float, 数值稳定性常数
        zeta: float, EOT正则化参数
        use_pytorch_adam: bool, 是否使用PyTorch自带的Adam优化器
        
        Returns
        -------
        X: torch.Tensor, 优化后的离散点坐标
        W: torch.Tensor, 优化后的离散点权重
        loss: float, 最终损失值
        L: float, 最终梯度范数
        """
        print(
            f"adam_discretize repeats={repeats}, use_pytorch_adam={use_pytorch_adam}"
        )

        # 确保输入是PyTorch张量并在正确的设备上
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = disc.clone().to(device)
        positions = positions.to(device)
        weights = weights.to(device)

        # 初始化目标点权重
        W = torch.ones(X.shape[0], device=device) / X.shape[0]

        if use_pytorch_adam:
            # 使用PyTorch自带的Adam优化器
            # 将X和W设置为需要梯度的参数
            X.requires_grad_(True)
            W.requires_grad_(True)

            # 创建优化器
            optimizer = torch.optim.Adam([X, W],
                                         lr=alpha,
                                         betas=(beta1, beta2),
                                         eps=epsilon)
            eps = 1e-5
            for i in range(repeats):
                # 清除之前的梯度
                optimizer.zero_grad()

                # 计算梯度和损失
                grad_W, grad_X, loss = self.grad(positions,
                                                 weights,
                                                 X,
                                                 W,
                                                 zeta=zeta)

                # 手动设置梯度
                if X.grad is None:
                    X.grad = torch.zeros_like(X)
                if W.grad is None:
                    W.grad = torch.zeros_like(W)

                X.grad.data = grad_X
                W.grad.data = grad_W

                # 执行优化步骤
                optimizer.step()

                # 限制X坐标在归一化范围[0, 1]
                with torch.no_grad():
                    X.data = torch.clamp(X.data, 0.0, 1.0)

                    # 保持W是一个合法的概率分布
                    W.data = torch.clamp(W.data, 1e-6, 1.0)
                    W.data = W.data / torch.sum(W.data)

                if i % 50 == 0 or i == repeats - 1:
                    with torch.no_grad():
                        L = torch.norm(grad_X) + torch.norm(grad_W)
                        print(
                            f"迭代 {i}, Loss: {loss.item():.6f}, Grad Norm: {L.item():.6f}"
                        )
                        if L < eps:
                            print("已收敛")
                            break
        else:
            # 手动实现Adam优化
            z_X = 0
            z_W = 0
            m_X = torch.zeros_like(X)  # 一阶矩估计
            v_X = torch.zeros_like(X)  # 二阶矩估计
            m_W = torch.zeros_like(W)  # 一阶矩估计
            v_W = torch.zeros_like(W)  # 二阶矩估计
            eps = 1e-5

            for i in range(repeats):
                # 计算梯度和损失
                grad_W, grad_X, loss = self.grad(positions,
                                                 weights,
                                                 X,
                                                 W,
                                                 zeta=zeta)

                # 更新一阶矩和二阶矩估计
                m_X = beta1 * m_X + (1 - beta1) * grad_X
                v_X = beta2 * v_X + (1 - beta2) * (grad_X**2)

                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * (grad_W**2)

                # 偏差修正
                m_X_hat = m_X / (1 - beta1**(i + 1))
                v_X_hat = v_X / (1 - beta2**(i + 1))

                m_W_hat = m_W / (1 - beta1**(i + 1))
                v_W_hat = v_W / (1 - beta2**(i + 1))

                # 使用 Adam 更新规则更新参数
                X -= alpha * m_X_hat / (torch.sqrt(v_X_hat) + epsilon)
                W -= alpha * m_W_hat / (torch.sqrt(v_W_hat) + epsilon)

                # 限制 X 坐标在归一化范围 [0, 1]，避免反归一化后越界
                X = torch.clamp(X, 0.0, 1.0)

                # 保持 W 是一个合法的概率分布
                W = torch.clamp(W, 1e-6, 1.0)
                W /= torch.sum(W)

                if i % 50 == 0 or i == repeats - 1:
                    L = torch.norm(grad_X) + torch.norm(grad_W)
                    print(
                        f"迭代 {i}, Loss: {loss.item():.6f}, Grad Norm: {L.item():.6f}"
                    )
                    if L < eps:
                        print("已收敛")
                        break

        # 确保返回的是CPU张量，便于后续处理
        return X.detach().cpu(), W.detach().cpu(), loss.item(), L.item()

    def discretize_from_coordinates(
            self,
            coord_points: np.ndarray,
            disc: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None,
            discrete_size: int = 4,
            repeats: int = 5000,
            use_pytorch_adam: bool = True,
            **args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        从坐标点离散化EDOT模型
                Parameters
        ----------
        coord_points: np.ndarray, 坐标点，形状为(n, 2)
        disc: Optional[np.ndarray], 初始离散点坐标，形状为(discrete_size, 2)
        weights: Optional[np.ndarray], 初始离散点权重，形状为(discrete_size)
        discrete_size: int, 离散点数量
        repeats: int, 最大迭代次数
        use_pytorch_adam: bool, 是否使用PyTorch自带的Adam优化器
        **args: 其他参数传递给adam_discretize
        
        Returns
        -------
        disc_rescaled: np.ndarray, 优化后的离散点坐标（反归一化后）
        weights: np.ndarray, 优化后的离散点权重
        labels: np.ndarray, 每个坐标点对应的离散点标签
        lastloss: float, 最终损失值
        lastL: float, 最终梯度范数
        """
        print(
            f"discretize_from_coordinates calls adam_discretize with repeats={repeats}"
        )
        print(f"discretize_size={discrete_size}")

        # 转换为numpy数组
        coord_points = np.asarray(coord_points, dtype=float)

        # 归一化坐标
        min_vals = coord_points.min(axis=0)
        max_vals = coord_points.max(axis=0)
        norm_coords = (coord_points - min_vals) / (max_vals - min_vals)

        # 转换为PyTorch张量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        norm_coords_torch = torch.tensor(norm_coords,
                                         dtype=torch.float32,
                                         device=device)
        wts = torch.ones(len(norm_coords), device=device) / len(norm_coords)

        # 初始化离散点
        if disc is None:
            disc_torch = torch.rand(discrete_size, 2,
                                    device=device) * 0.9 + 0.05
            disc_torch += torch.rand(discrete_size, 2, device=device) * 1e-1
        else:
            disc_torch = torch.tensor(disc, dtype=torch.float32, device=device)

        if weights is None:
            weights_torch = torch.ones(discrete_size,
                                       device=device) / discrete_size
        else:
            weights_torch = torch.tensor(weights,
                                         dtype=torch.float32,
                                         device=device)

        # 调用Adam优化
        disc_torch, weights_torch, lastloss, lastL = self.adam_discretize(
            disc_torch,
            norm_coords_torch,
            wts,
            repeats=repeats,
            use_pytorch_adam=use_pytorch_adam,
            **args)

        # 反归一化
        disc_rescaled = disc_torch.numpy() * (max_vals - min_vals) + min_vals

        # 计算标签
        distances = np.zeros((len(coord_points), discrete_size))
        for i in range(len(coord_points)):
            for j in range(discrete_size):
                distances[i, j] = np.sum(
                    (coord_points[i] - disc_rescaled[j])**2)
        labels = np.argmin(distances, axis=1)

        return disc_rescaled, weights_torch.numpy(), labels, lastloss, lastL


def main():
    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)

    sheetnames = ['12-1']
    for sheet in sheetnames:
        print(f"Processing sheet: {sheet}")

        df = pd.read_excel("E:/behavioral data/eyetracking_faceid.xlsx",
                           sheet_name=sheet)
        trialcol = df['trial'].values
        trials = np.unique(trialcol)

        for tri in trials[6:9]:
            print(f"Processing trial {tri}")
            df_trial = df[df['trial'] == tri]
            if len(df_trial) < 10:
                print(f"Skipping trial {tri} due to insufficient data")
                continue

            x = df_trial['x_position'].values
            y = df_trial['y_position'].values
            eye_movements = np.column_stack((x, y))

            edot_model = EDOT_Torch()
            discrete_sizes = range(3, 4)

            for size in discrete_sizes:
                print(f"\n=== 处理 discrete_size = {size} ===")

                # 使用PyTorch Adam优化器
                centers, weights, labels, lastloss, lastL = edot_model.discretize_from_coordinates(
                    eye_movements,
                    discrete_size=size,
                    repeats=5000,
                    use_pytorch_adam=True)

                # 准备结果字典
                trial_result = {
                    size: {
                        "centers": centers.tolist(),
                        "weights": weights.tolist(),
                        "loss": float(lastloss),
                        "final_gradient": float(lastL)
                    }
                }

                # 保存 JSON 文件（每个 trial 单独保存）
                json_path = f"edot_results_adam_torch/{sheet}_edot_summary_trial_{int(tri)}_size_{int(size)}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(trial_result, f, ensure_ascii=False, indent=2)

                print(f"✅ Trial {tri} 的结果已保存为 {json_path}")

                # 可视化
                colors = plt.cm.get_cmap('tab20', size)(
                    labels % 20)  # 防止标签数超过 colormap 范围

                plt.figure(figsize=(19.2, 10.8), dpi=100)
                image_path = "E:/behavioral data/image/task_see/EF_gt_185.png"
                if os.path.exists(image_path):
                    img = plt.imread(image_path)
                    plt.imshow(img, extent=[0, 1920, 1080, 0])
                else:
                    print(f"⚠️ 图像文件未找到: {image_path}, 使用空背景")

                plt.scatter(x,
                            y,
                            c=colors,
                            alpha=0.7,
                            label='eyemovements',
                            zorder=1)

                # 根据权重设置中心点大小
                plt.scatter(centers[:, 0],
                            centers[:, 1],
                            c='red',
                            s=weights * 2000,
                            marker='X',
                            label='centers',
                            zorder=2)

                for i, center in enumerate(centers):
                    plt.text(center[0],
                             center[1],
                             f'{i+1}',
                             fontsize=12,
                             ha='center',
                             va='center',
                             bbox=dict(facecolor='white',
                                       alpha=0.8,
                                       edgecolor='none'),
                             zorder=3)

                plt.title(
                    f"EDOT result (regions={size}), Trial {tri}, lastL={lastL:.5f}"
                )
                plt.xlabel("X (pixels)")
                plt.ylabel("Y (pixels)")
                plt.xlim(0, 1920)
                plt.ylim(1080, 0)
                plt.legend()
                plt.grid(False)

                img_save_path = f"edot_results_adam_torch/{sheet}_trial{tri}_clustering_size_{size}.png"
                plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"🖼️ Trial {tri} 的聚类图已保存为 {img_save_path}")


if __name__ == "__main__":
    main()

__all__ = ['EDOT_Torch']
