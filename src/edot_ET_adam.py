import numpy as np
import matplotlib.pyplot as plt
from . import EDOT_discretization  # 确保存在
from .EDOT_discretization import gradient_POS
import pandas as pd
import os
from scipy.spatial.distance import cdist
import json

# os.chdir("E:/behavioral data/eyemovement markov algorithm")  # 切换到上级目录以确保相对导入工作
os.makedirs("edot_results_adam", exist_ok=True)


class EDOT:

    def __init__(self):
        pass

    def adam_descretize(self,
                        disc,
                        positions,
                        weights,
                        repeats=5000,
                        alpha=0.005,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-8,
                        zeta=0.004):
        print(f"adam_descretize repeats={repeats}")

        X = disc.copy()
        W = np.ones(X.shape[0]) / X.shape[0]  # 初始化目标点权重
        z_X = 0
        z_W = 0
        m_X = np.zeros_like(X)  # 一阶矩估计
        v_X = np.zeros_like(X)  # 二阶矩估计
        m_W = np.zeros_like(W)  # 一阶矩估计
        v_W = np.zeros_like(W)  # 二阶矩估计
        eps = 1e-5

        for i in range(repeats):
            grad_X, loss, grad_W = gradient_POS(positions,
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
            X -= alpha * m_X_hat / (np.sqrt(v_X_hat) + epsilon)
            W -= alpha * m_W_hat / (np.sqrt(v_W_hat) + epsilon)

            # ✅ 限制 X 坐标在归一化范围 [0, 1]，避免反归一化后越界
            X = np.clip(X, 0.0, 1.0)

            # 保持 W 是一个合法的概率分布
            W = np.clip(W, 1e-6, 1.0)
            W /= np.sum(W)

            if i % 50 == 0 or i == repeats - 1:
                L = np.linalg.norm(grad_X) + np.linalg.norm(grad_W)
                print(f"迭代 {i}, Loss: {loss:.6f}, Grad Norm: {L:.6f}")
                if L < eps:
                    print("已收敛")
                    break

        return X, W, loss, L

    def discretize_from_coordinates(self,
                                    coord_points,
                                    disc=None,
                                    weights=None,
                                    discrete_size=4,
                                    repeats=5000,
                                    **args):
        print(
            f"discretize_from_coordinates calls adam_descretize with repeats={repeats}"
        )
        print(f"discretize_size={discrete_size}")

        coord_points = np.asarray(coord_points, dtype=float)

        min_vals = coord_points.min(axis=0)
        max_vals = coord_points.max(axis=0)
        norm_coords = (coord_points - min_vals) / (max_vals - min_vals)

        wts = np.ones(len(norm_coords)) / len(norm_coords)

        if disc is None:
            disc = np.random.rand(discrete_size, 2)
        if weights is None:
            weights = np.ones(discrete_size) / discrete_size

        disc = np.random.rand(discrete_size, 2) * 0.9 + 0.05
        disc += np.random.rand(*disc.shape) * 1e-1

        disc, weights, lastloss, lastL = self.adam_descretize(disc,
                                                              norm_coords,
                                                              wts,
                                                              repeats=repeats,
                                                              **args)

        disc_rescaled = disc * (max_vals - min_vals) + min_vals

        distances = cdist(coord_points, disc_rescaled)
        labels = np.argmin(distances, axis=1)

        return disc_rescaled, weights, labels, lastloss, lastL


def main():
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

            edot_model = EDOT()
            discrete_sizes = range(3, 4)

            for size in discrete_sizes:
                print(f"\n=== 处理 discrete_size = {size} ===")

                centers, weights, labels, lastloss, lastL = edot_model.discretize_from_coordinates(
                    eye_movements, discrete_size=size, repeats=5000)

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
                json_path = f"edot_results_adam/{sheet}_edot_summary_trial_{int(tri)}_size_{int(size)}.json"
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

                img_save_path = f"edot_results_adam/{sheet}_trial{tri}_clustering_size_{size}.png"
                plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"🖼️ Trial {tri} 的聚类图已保存为 {img_save_path}")


if __name__ == "__main__":
    main()
