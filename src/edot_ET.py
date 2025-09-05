import numpy as np
import matplotlib.pyplot as plt
from . import EDOT_discretization  # 请确保此函数已定义
from .EDOT_discretization import gradient_POS  # 确保导入正确
import pandas as pd
import os
from scipy.spatial.distance import cdist

# 创建保存结果的目录
os.makedirs("edot_results", exist_ok=True)

class EDOT:
    def __init__(self):
        pass

    def lagrange_descretize(self, disc, positions, weights,
                          repeats=1000, alpha=0.01, beta=0.95, zeta=0.004):
        print(f"lagrange_descretize repeats={repeats}")
        
        X = disc.copy()
        W = np.ones(X.shape[0]) / X.shape[0]
        z = 0
        eps = 1e-5

        for i in range(repeats):
            grad = gradient_POS(positions, weights, X, W, zeta=zeta)
            z = beta * z + grad[0]
            X -= alpha * z

            if i % 50 == 0 or i == repeats - 1:
                L = np.linalg.norm(grad[0])
                print(f"迭代 {i}, 梯度范数: {L:.6f}")
                if L < eps:
                    print("已收敛")
                    break
        return X, L

    def discretize_from_coordinates(self, coord_points, disc=None, weights=None,
                                 discrete_size=4, repeats=1000, **args):
        print(f"discretize_from_coordinates calls lagrange_descretize with repeats={repeats}")
        print(f"discretize_size={discrete_size}")
        
        coord_points = np.asarray(coord_points, dtype=float)
        
        # Min-max 归一化
        min_vals = coord_points.min(axis=0)
        max_vals = coord_points.max(axis=0)
        norm_coords = (coord_points - min_vals) / (max_vals - min_vals)

        # 均匀权重
        wts = np.ones(len(norm_coords)) / len(norm_coords)

        # 初始化中心点
        if disc is None:
            disc = np.random.rand(discrete_size, 2)
        if weights is None:
            weights = np.ones(discrete_size) / discrete_size

        disc = np.random.rand(discrete_size, 2) * 0.9 + 0.05
        disc += np.random.rand(*disc.shape) * 1e-1

        # 执行优化
        disc, lastL = self.lagrange_descretize(
            disc, norm_coords, wts, repeats=repeats, **args)

        # 反归一化中心点坐标
        disc_rescaled = disc * (max_vals - min_vals) + min_vals
        
        # 计算每个点所属的分区
        distances = cdist(coord_points, disc_rescaled)
        labels = np.argmin(distances, axis=1)
        
        return disc_rescaled, weights, labels, lastL

def main():
    # 加载数据
    df = pd.read_excel("E:/behavioral data/eyetracking_faceid.xlsx", sheet_name='18-2')
    trialcol = df['trial'].values
    trials = np.unique(trialcol)
    for tri in trials:
        print(f"Processing trial {tri}")
        df_trial = df[df['trial'] == tri]
        if len(df_trial) < 10:
            print(f"Skipping trial {tri} due to insufficient data")
            continue
        x = df_trial['x_position'].values
        y = df_trial['y_position'].values
        eye_movements = np.column_stack((x, y))

        edot_model = EDOT()
        discrete_sizes = range(3, 13)  # 3到12

        for size in discrete_sizes:
            print(f"\n=== 处理 discrete_size = {size} ===")
            
            # 运行EDOT算法
            centers, _, labels, lastL = edot_model.discretize_from_coordinates(
                eye_movements,
                discrete_size=size,
                repeats=1000
            )

            # 保存中心点坐标
            np.savetxt(f"edot_results/trial{tri}_centers_size_{size}.txt", centers, fmt="%.2f")
            
            # 创建彩色标签
            colors = plt.cm.get_cmap('tab20', size)(labels)
            
            # 可视化
            plt.figure(figsize=(19.2, 10.8), dpi=100)  # 模拟屏幕 1920×1080 分辨率

            # 读取背景图像（1920x1080）
            img = plt.imread("E:/behavioral data/image/task_see/EF_gt_185.png")  # 替换成你自己的路径

            # 显示图像，坐标范围设定为图像尺寸，确保图像不会被压缩
            plt.imshow(img, extent=[0, 1920, 1080, 0])  # 注意 y 倒置，符合屏幕坐标（左上角为原点）

            # 叠加眼动点
            plt.scatter(x, y, c=colors, alpha=0.7, label='eyemovements', zorder=1)

            # 叠加中心点
            plt.scatter(centers[:, 0], centers[:, 1], 
                        c='red', s=200, marker='X', label='centers', zorder=2)

            # 添加中心编号
            for i, center in enumerate(centers):
                plt.text(center[0], center[1], f'{i+1}',
                        fontsize=12, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=3)

            plt.title(f"EDOT clustering result (number of region={size}), Trial {tri}, Last L={lastL:.5f}")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.xlim(0, 1920)
            plt.ylim(1080, 0)  # 屏幕坐标，(0,0) 是左上角
            plt.legend()
            plt.grid(False)

            # 保存图像
            plt.savefig(f"edot_results/trial{tri}_clustering_size_{size}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"结果已保存到 edot_results/ 目录")

if __name__ == "__main__":
    main()