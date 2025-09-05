from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from . import EDOT_discretization  # 确保存在
from .EDOT_discretization import gradient_POS, gradient_EOT
import pandas as pd
import os
from scipy.spatial.distance import cdist
import json

os.makedirs("edot_results_new", exist_ok=True)


class EDOT:

    def __init__(self):
        pass

    def EDOT_descretize(self,
                        disc,
                        positions,
                        weights,
                        repeats=5000,
                        alpha=0.05,
                        beta=0.95,
                        zeta=0.004):
        print(f"EDOT_descretize repeats={repeats}")

        X = disc.copy()
        W = np.ones(X.shape[0]) / X.shape[0]  # 初始化目标点权重
        z_X = 0
        z_W = 0
        eps = 1e-5

        for i in range(repeats):
            grad_W, grad_X, loss = gradient_EOT(positions,
                                                weights,
                                                X,
                                                W,
                                                zeta=zeta)

            # Momentum 更新
            z_X = beta * z_X + grad_X
            z_W = beta * z_W + grad_W

            X -= alpha * z_X
            W -= alpha * z_W

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
                                    repeats=1000,
                                    **args):
        print("discretize_from_coordinates calls EDOT_descretize"
              f" with repeats={repeats}")
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

        (disc, weights, lastloss,
         lastL) = self.EDOT_descretize(disc,
                                       norm_coords,
                                       wts,
                                       repeats=repeats,
                                       **args)

        disc_rescaled = disc * (max_vals - min_vals) + min_vals

        distances = cdist(coord_points, disc_rescaled)
        labels = np.argmin(distances, axis=1)

        return disc_rescaled, weights, labels, lastloss, lastL


def main(root: Path, **kwargs):
    sheetnames = ['12-1', '14-1']
    for sheet in sheetnames:
        print(f"Processing sheet: {sheet}")

        df = pd.read_excel(root / "eyetracking_faceid.xlsx", sheet_name=sheet)
        trialcol = df['trial'].values
        trials = np.unique(trialcol)

        for tri in trials[14:]:
            print(f"Processing trial {tri}")
            df_trial = df[df['trial'] == tri]
            if len(df_trial) < 10:
                print(f"Skipping trial {tri} due to insufficient data")
                continue

            x = df_trial['x_position'].values
            y = df_trial['y_position'].values

            eye_movements = np.column_stack((x, y))
            print(eye_movements, np.max(eye_movements, axis=0),
                  np.min(eye_movements, axis=1))
            edot_model = EDOT()
            discrete_sizes = range(3, 13)
            affine_ratio = max(
                np.max(eye_movements, axis=0) - np.min(eye_movements, axis=0))
            affine_offset = np.min(eye_movements, axis=0)
            eye_move = (eye_movements - affine_offset) / affine_ratio
            for size in discrete_sizes:
                print(f"\n=== 处理 discrete_size = {size} ===")

                (centers, weights, labels, lastloss,
                 lastL) = edot_model.discretize_from_coordinates(
                     eye_move, discrete_size=size, repeats=5000, **kwargs)
                print(centers, weights)
                centers = centers * affine_ratio + affine_offset
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
                json_path = f"edot_results_new/{sheet}_edot_summary_trial_{int(tri)}_size_{int(size)}.json"

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(trial_result, f, ensure_ascii=False, indent=2)

                print(f"✅ Trial {tri} 的结果已保存为 {json_path}")

                # 可视化
                colors = plt.cm.get_cmap('tab20', size)(labels % 20)
                # 防止标签数超过 colormap 范围

                plt.figure(figsize=(19.2, 10.8), dpi=100)
                image_path = root / "image/task_see/EF_gt_185.png"
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
                plt.text(0,
                         0,
                         "\n".join([str(x) for x in weights]),
                         fontsize=12,
                         ha='left',
                         va='top',
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

                img_save_path = f"edot_results_new/{sheet}_trial{tri}_clustering_size_{size}.png"
                plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"🖼️ Trial {tri} 的聚类图已保存为 {img_save_path}")


if __name__ == "__main__":
    main()
