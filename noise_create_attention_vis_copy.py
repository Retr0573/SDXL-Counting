"""
noise更改测试

根据attention map的高响应部分来创造noise
"""

from math import atan
from utils.attention_sdxl_pipeline import AttentionStableDiffusionXLPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import torch
import os
import inflect
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import label
import cv2
from skimage.feature import peak_local_max
from skimage import io, filters, measure
from skimage.segmentation import watershed
from scipy.ndimage import binary_erosion
from scipy import ndimage as ndi

p = inflect.engine()

PIPELINE_PATH = "/data/wsq_data/stable-diffusion-xl-base-1/"

DEVICE = "cuda:0"


def create_prompt_list():
    numbers_list = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
    ]
    objects_list = [
        "giraffe",
        "zebra",
        "bear",
        "cow",
        "sheep",
        "horse",
        "dog",
        "cat",
        "bird",
        "apple",
        "donut",
        "bowl",
        "cup",
        "car",
        "backpack",
    ]
    prompts_list = []
    object_list = []
    num_int_list = []
    for i, num in enumerate(numbers_list):
        num_int = i + 1
        for obj in objects_list:
            plural_obj = obj if num == "one" else p.plural(obj)
            prompts_list.append(f"a photo of {num} {plural_obj}")
            object_list.append(obj)
            num_int_list.append(num_int)
    return prompts_list, object_list, num_int_list


def noise_recreate_1(latents, attention_map, numeral):
    """
    基于attention map的空间分布特征重组latents
    1. 使用聚类算法（K-mean）找出attention高响应区域（更倾向于根据位置平均分布）
    2. 确保这些区域在空间上合理分布
    3. 在目标区域周围创建平滑的noise增强/减弱mask
    """
    import numpy as np
    from sklearn.cluster import KMeans

    # 1. 准备attention map数据
    h, w = attention_map.shape
    coords = np.array([(i, j) for i in range(h) for j in range(w)])
    values = attention_map.flatten()

    # 2. 使用K-means聚类找出numeral个中心点
    kmeans = KMeans(n_clusters=numeral, random_state=42)
    weights = values / values.sum()  # 使用attention值作为权重
    kmeans.fit(coords, sample_weight=weights)
    centers = kmeans.cluster_centers_

    # 3. 计算每个点到最近中心的距离
    dist_map = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            dist_map[i, j] = np.min(np.linalg.norm(centers - [i, j], axis=1))

    # 4. 创建基于距离的增强mask
    max_dist = dist_map.max()
    mask = 1.0 - 1.8 * (dist_map / max_dist)  # 中心区域值接近1，边缘接近0
    mask = np.clip(mask * 2.0, 0.9, 1.1)  # 控制增强范围

    # 5. 应用mask到latents
    mask = torch.from_numpy(mask).to(latents.device, dtype=latents.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 匹配latents的维度
    latents = latents * mask

    return latents


def noise_recreate_2(
    latents, attention_map, numeral, alpha=10, save_dir="./vis_output", prefix="sample"
):
    """
    基于attention map的空间分布特征重组latents
    1. 使用聚类算法（K-mean）找出attention高响应区域，注意CA值作为kmeans考虑的weight，应加大其分化度来提高其影响
    2. 确保这些区域在空间上合理分布
    3. 在目标区域周围创建平滑的noise增强/减弱mask，用更合理的递减，为每个像素分配到一个mask，该mask内值从0.9-1.1
    """

    def build_nonoverlapping_mask(h, w, centers, save_dir=None, prefix=None):
        numeral = len(centers)

        # 1. 计算每个像素到各个中心的欧几里得距离
        dist_tensor = np.zeros((numeral, h, w), dtype=np.float32)
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        for idx, (cy, cx) in enumerate(centers):
            dist_tensor[idx] = np.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)

        # 2. 每个像素归属最近的中心
        closest_center = np.argmin(dist_tensor, axis=0)

        # 3. 使用 BFS 确定每个中心的最大“安全步数”
        owner = -np.ones((h, w), dtype=np.int32)
        step_map = np.full((h, w), np.inf, dtype=np.float32)
        queue = deque()
        min_dist = [9999999] * numeral

        for idx, (cy, cx) in enumerate(centers):
            queue.append((cy, cx, idx, 0))
            owner[cy, cx] = idx
            step_map[cy, cx] = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            y, x, center_id, step = queue.popleft()

            for dy, dx in directions:
                ny, nx = y + dy, x + dx

                # 情况 1：越界，记录当前 step 作为到图像边界的最短距离
                if not (0 <= ny < h and 0 <= nx < w):
                    min_dist[center_id] = min(min_dist[center_id], step)
                    continue

                # 情况 2：首次访问，继续扩展
                if owner[ny, nx] == -1:
                    owner[ny, nx] = center_id
                    step_map[ny, nx] = step + 1
                    queue.append((ny, nx, center_id, step + 1))

                # 情况 3：遇到其他中心的区域，记录当前 step 作为“最近边界”
                elif owner[ny, nx] != center_id:
                    min_dist[center_id] = min(min_dist[center_id], step)

        # 4. 构建最终 mask：在 BFS 确定的最大半径范围内，从中心递减
        mask = np.ones((h, w), dtype=np.float32) * 1
        for idx in range(numeral):
            region = closest_center == idx
            if not region.any():
                continue

            dist = dist_tensor[idx]
            max_safe_dist = min_dist[idx]
            # 限制最大范围：不能超过 bfs 给出的安全步长

            region_mask = 1.2 - 0.2 * (dist / max_safe_dist)
            region_mask[region_mask < 1] = 0.95
            region_mask = np.clip(region_mask, 0.9, 1.2)
            mask[region] = region_mask[region]

            plt.figure(figsize=(6, 6))
            plt.imshow(region_mask, cmap="coolwarm")
            plt.title("Mask")
            plt.colorbar()
            plt.axis("off")
            plt.savefig(
                os.path.join(save_dir, f"{prefix}_mask_{idx}.png"), bbox_inches="tight"
            )
            plt.close()

        return mask

    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    os.makedirs(save_dir, exist_ok=True)

    # 1. 准备attention map数据
    h, w = attention_map.shape
    coords = np.array([(i, j) for i in range(h) for j in range(w)])
    values = attention_map.flatten()

    # 2. 使用K-means聚类找出numeral个中心点
    kmeans = KMeans(n_clusters=numeral, random_state=42)
    # weights = values / values.sum()
    # weights = values**alpha
    weights = np.exp(values * alpha)
    weights = weights / weights.sum()
    kmeans.fit(coords, sample_weight=weights)
    centers = kmeans.cluster_centers_
    centers = np.round(centers).astype(int)
    # # 3. 计算每个点到最近中心的距离
    # dist_map = np.zeros((h, w))
    # for i in range(h):
    #     for j in range(w):
    #         dist_map[i, j] = np.min(np.linalg.norm(centers - [i, j], axis=1))

    # # 4. 创建基于距离的增强mask
    # max_dist = dist_map.max()
    # mask = 1.0 - 1.8 * (dist_map / max_dist)
    # mask = np.clip(mask * 2.0, 0.9, 1.1)

    mask = build_nonoverlapping_mask(
        h=h, w=w, centers=centers, save_dir=save_dir, prefix=prefix
    )

    # 5. 可视化 attention_map + centers
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap="viridis")
    plt.scatter(
        centers[:, 1], centers[:, 0], c="red", marker="x", s=50, label="Centers"
    )
    plt.title("Attention Map with Centers")
    plt.legend()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_attention_centers.png"), bbox_inches="tight"
    )
    plt.close()

    # 6. 可视化 mask
    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="coolwarm")
    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{prefix}_mask.png"), bbox_inches="tight")
    plt.close()

    # 7. 也可视化原始 attention_map
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap="viridis")
    plt.title("Original Attention Map")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_attention_map.png"), bbox_inches="tight"
    )
    plt.close()

    # 8. 应用 mask 到 latents
    mask = torch.from_numpy(mask).to(latents.device, dtype=latents.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 匹配 latents 的维度
    latents = latents * mask

    return latents


def noise_recreate_2_2(
    latents, attention_map, numeral, alpha=10, save_dir="./vis_output", prefix="sample"
):
    """
    在v2-original基础上，修改了mask的值分布
    基于attention map的空间分布特征重组latents
    1. 使用聚类算法（K-mean）找出attention高响应区域，注意CA值作为kmeans考虑的weight，应加大其分化度来提高其影响
    2. 确保这些区域在空间上合理分布
    3. 在目标区域周围创建平滑的noise增强/减弱mask，为每个像素分配到一个mask，该mask内默认为1.1， edge部分为0.9， background部分为0.95
    """

    def build_nonoverlapping_mask(
        h,
        w,
        centers,
        alpha_content=1.1,
        alpha_edge=0.9,
        alpha_background=0.95,
        save_dir=None,
        prefix=None,
    ):
        numeral = len(centers)

        # 1. 计算每个像素到各个中心的欧几里得距离
        dist_tensor = np.zeros((numeral, h, w), dtype=np.float32)
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        for idx, (cy, cx) in enumerate(centers):
            dist_tensor[idx] = np.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)

        # 2. 每个像素归属最近的中心
        closest_center = np.argmin(dist_tensor, axis=0)

        # 3. 使用 BFS 确定每个中心的最大“安全步数”
        owner = -np.ones((h, w), dtype=np.int32)
        step_map = np.full((h, w), np.inf, dtype=np.float32)
        queue = deque()
        min_dist = [9999999] * numeral

        for idx, (cy, cx) in enumerate(centers):
            queue.append((cy, cx, idx, 0))
            owner[cy, cx] = idx
            step_map[cy, cx] = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            y, x, center_id, step = queue.popleft()

            for dy, dx in directions:
                ny, nx = y + dy, x + dx

                # 情况 1：越界，记录当前 step 作为到图像边界的最短距离
                if not (0 <= ny < h and 0 <= nx < w):
                    min_dist[center_id] = min(min_dist[center_id], step)
                    continue

                # 情况 2：首次访问，继续扩展
                if owner[ny, nx] == -1:
                    owner[ny, nx] = center_id
                    step_map[ny, nx] = step + 1
                    queue.append((ny, nx, center_id, step + 1))

                # 情况 3：遇到其他中心的区域，记录当前 step 作为“最近边界”
                elif owner[ny, nx] != center_id:
                    min_dist[center_id] = min(min_dist[center_id], step)

        # 4. 构建最终 mask：每个中心为1.2，边缘为0.9，之间为1.2
        mask = np.ones((h, w), dtype=np.float32)
        max_safe_radius = h / max(numeral, 3) / 2  # 设置 min_dist 上限
        edge_width = 3  # 你可以改这个厚度，单位是像素

        for idx in range(numeral):
            region = closest_center == idx
            if not region.any():
                continue

            dist = dist_tensor[idx]
            safe_dist = min(min_dist[idx], max_safe_radius)

            # 定义 mask：默认 1.2
            region_mask = np.ones_like(dist) * alpha_background

            # 找出边缘区域：在 [safe_dist - edge_width, safe_dist] 范围内的
            edge_mask = (dist >= (safe_dist - edge_width)) & (dist <= safe_dist)
            content_mask = dist < safe_dist - edge_width
            region_mask[edge_mask] = alpha_edge
            region_mask[content_mask] = alpha_content
            # 将 mask 应用于对应区域
            mask[region] = region_mask[region]

            # plt.figure(figsize=(6, 6))
            # plt.imshow(region_mask, cmap="coolwarm")
            # plt.title("Mask")
            # plt.colorbar()
            # plt.axis("off")
            # plt.savefig(
            #     os.path.join(save_dir, f"{prefix}_mask_{idx}.png"), bbox_inches="tight"
            # )
            # plt.close()

        return mask

    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    os.makedirs(save_dir, exist_ok=True)

    # 1. 准备attention map数据
    h, w = attention_map.shape
    coords = np.array([(i, j) for i in range(h) for j in range(w)])
    values = attention_map.flatten()

    # 2. 使用K-means聚类找出numeral个中心点
    kmeans = KMeans(n_clusters=numeral, random_state=42)
    # weights = values / values.sum()
    # weights = values**alpha
    weights = np.exp(values * alpha)
    weights = weights / weights.sum()
    kmeans.fit(coords, sample_weight=weights)
    centers = kmeans.cluster_centers_
    centers = np.round(centers).astype(int)

    # 3. 创造mask
    mask = build_nonoverlapping_mask(
        h=h, w=w, centers=centers, save_dir=save_dir, prefix=prefix
    )

    # 4. 可视化 attention_map + centers
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap="viridis")
    plt.scatter(
        centers[:, 1], centers[:, 0], c="red", marker="x", s=50, label="Centers"
    )
    plt.title("Attention Map with Centers")
    plt.legend()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_attention_centers.png"), bbox_inches="tight"
    )
    plt.close()

    # 5. 可视化 mask
    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="coolwarm")
    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{prefix}_mask.png"), bbox_inches="tight")
    plt.close()

    # # 7. 也可视化原始 attention_map
    # plt.figure(figsize=(6, 6))
    # plt.imshow(attention_map, cmap="viridis")
    # plt.title("Original Attention Map")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(
    #     os.path.join(save_dir, f"{prefix}_attention_map.png"), bbox_inches="tight"
    # )
    # plt.close()

    # 8. 应用 mask 到 latents
    mask = torch.from_numpy(mask).to(latents.device, dtype=latents.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 匹配 latents 的维度
    latents = latents * mask

    return latents


def noise_recreate_2_3(
    latents,
    attention_map,
    numeral,
    alpha=10,
    alpha_content=1.15,
    alpha_edge=0.9,
    alpha_background=0.95,
    save_dir="./vis_output",
    prefix="sample",
):
    """
    在v2-2的基础上，修改mask中，具体mask圆的大小和内部值的大小
    基于attention map的空间分布特征重组latents
    1. 使用聚类算法（K-mean）找出attention高响应区域，注意CA值作为kmeans考虑的weight，应加大其分化度来提高其影响
    2. 确保这些区域在空间上合理分布
    3. 在目标区域周围创建平滑的noise增强/减弱mask，为每个像素分配到一个mask，该mask内默认为1.15， edge部分为0.9， background部分为0.95
    """

    def build_nonoverlapping_mask(
        h,
        w,
        centers,
        alpha_content=1.15,
        alpha_edge=0.9,
        alpha_background=0.95,
        save_dir=None,
        prefix=None,
    ):
        numeral = len(centers)

        # 1. 计算每个像素到各个中心的欧几里得距离
        dist_tensor = np.zeros((numeral, h, w), dtype=np.float32)
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        for idx, (cy, cx) in enumerate(centers):
            dist_tensor[idx] = np.sqrt((grid_y - cy) ** 2 + (grid_x - cx) ** 2)

        # 2. 每个像素归属最近的中心
        closest_center = np.argmin(dist_tensor, axis=0)

        # 3. 使用 BFS 确定每个中心的最大“安全步数”
        owner = -np.ones((h, w), dtype=np.int32)
        step_map = np.full((h, w), np.inf, dtype=np.float32)
        queue = deque()
        min_dist = [9999999] * numeral

        for idx, (cy, cx) in enumerate(centers):
            queue.append((cy, cx, idx, 0))
            owner[cy, cx] = idx
            step_map[cy, cx] = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            y, x, center_id, step = queue.popleft()

            for dy, dx in directions:
                ny, nx = y + dy, x + dx

                # 情况 1：越界，记录当前 step 作为到图像边界的最短距离
                if not (0 <= ny < h and 0 <= nx < w):
                    min_dist[center_id] = min(min_dist[center_id], step)
                    continue

                # 情况 2：首次访问，继续扩展
                if owner[ny, nx] == -1:
                    owner[ny, nx] = center_id
                    step_map[ny, nx] = step + 1
                    queue.append((ny, nx, center_id, step + 1))

                # 情况 3：遇到其他中心的区域，记录当前 step 作为“最近边界”
                elif owner[ny, nx] != center_id:
                    min_dist[center_id] = min(min_dist[center_id], step)

        # 4. 构建最终 mask：每个中心为1.2，边缘为0.9，之间为1.2
        mask = np.ones((h, w), dtype=np.float32)
        max_safe_radius = (
            h / max(numeral, 3) / 2 * 1.5
        )  # 设置 min_dist 上限，以防mask中具体的mask圆过大；这里最后乘以1.5，是希望mask的半径可以更大些，这更适合numeral比较大的时候，防止mask圆过小
        edge_width = 3  # 你可以改这个厚度，单位是像素
        edge_width = 3  # 你可以改这个厚度，单位是像素

        for idx in range(numeral):
            region = closest_center == idx
            if not region.any():
                continue

            dist = dist_tensor[idx]
            safe_dist = min(min_dist[idx], max_safe_radius)

            # 定义 mask：默认 1.2
            region_mask = np.ones_like(dist) * alpha_background

            # 找出边缘区域：在 [safe_dist - edge_width, safe_dist] 范围内的
            edge_mask = (dist >= (safe_dist - edge_width)) & (dist <= safe_dist)
            content_mask = dist < safe_dist - edge_width
            region_mask[edge_mask] = alpha_edge
            region_mask[content_mask] = alpha_content
            # 将 mask 应用于对应区域
            mask[region] = region_mask[region]

            # plt.figure(figsize=(6, 6))
            # plt.imshow(region_mask, cmap="coolwarm")
            # plt.title("Mask")
            # plt.colorbar()
            # plt.axis("off")
            # plt.savefig(
            #     os.path.join(save_dir, f"{prefix}_mask_{idx}.png"), bbox_inches="tight"
            # )
            # plt.close()

        return mask

    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    os.makedirs(save_dir, exist_ok=True)

    # 1. 准备attention map数据
    h, w = attention_map.shape
    coords = np.array([(i, j) for i in range(h) for j in range(w)])
    values = attention_map.flatten()

    # 2. 使用K-means聚类找出numeral个中心点
    kmeans = KMeans(n_clusters=numeral, random_state=42)
    # weights = values / values.sum()
    # weights = values**alpha
    weights = np.exp(values * alpha)
    weights = weights / weights.sum()
    kmeans.fit(coords, sample_weight=weights)
    centers = kmeans.cluster_centers_
    centers = np.round(centers).astype(int)

    # 3. 创造mask
    mask = build_nonoverlapping_mask(
        h=h,
        w=w,
        centers=centers,
        alpha_content=alpha_content,
        alpha_edge=alpha_edge,
        alpha_background=alpha_background,
        save_dir=save_dir,
        prefix=prefix,
    )

    # 4. 可视化 attention_map + centers
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap="viridis")
    plt.scatter(
        centers[:, 1], centers[:, 0], c="red", marker="x", s=50, label="Centers"
    )
    plt.title("Attention Map with Centers")
    plt.legend()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_attention_centers.png"), bbox_inches="tight"
    )
    plt.close()

    # 5. 可视化 mask
    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="coolwarm")
    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{prefix}_mask.png"), bbox_inches="tight")
    plt.close()

    # # 7. 也可视化原始 attention_map
    # plt.figure(figsize=(6, 6))
    # plt.imshow(attention_map, cmap="viridis")
    # plt.title("Original Attention Map")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(
    #     os.path.join(save_dir, f"{prefix}_attention_map.png"), bbox_inches="tight"
    # )
    # plt.close()

    # 8. 应用 mask 到 latents
    mask = torch.from_numpy(mask).to(latents.device, dtype=latents.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 匹配 latents 的维度
    latents = latents * mask

    return latents


def noise_recreate_3(
    latents,
    attention_map,
    numeral,
    strength=1.2,
    threshold_percent=93,
    save_dir="./vis_output",
    prefix="sample",
):
    """
    基于cross-attention map的空间分布特征来创建mask，目标是找到numeral个attention汇聚的区域，然后在mask中给这些区域提高value
    """
    # -------------------------
    # 准备attention map数据
    # -------------------------
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    attention_map_np = (
        attention_map.cpu().numpy()
        if isinstance(attention_map, torch.Tensor)
        else attention_map
    )
    # -------------------------
    # 创建mask
    # -------------------------
    # 阈值化：提取高响应区域
    threshold = np.percentile(attention_map_np, threshold_percent)
    binary_map = (attention_map_np > threshold).astype(np.uint8)

    # 连通区域标记
    labeled_map, num_features = label(binary_map)

    # 如果区域少于 numeral，就降阈值再提取
    while num_features < numeral and threshold > 0.1:
        threshold *= 0.9
        binary_map = (attention_map_np > threshold).astype(np.uint8)
        labeled_map, num_features = label(binary_map)

    # 找出所有区域的最大attention点
    centers = []
    for label_id in range(1, num_features + 1):
        region_mask = labeled_map == label_id
        if np.any(region_mask):
            region_values = attention_map_np * region_mask
            max_idx = np.unravel_index(np.argmax(region_values), attention_map_np.shape)
            max_val = attention_map_np[max_idx]
            centers.append((max_idx, max_val))  # max_idx 是 (x, y)
    # 对每个区域的 最大attention 值排序，选 top numeral 个
    top_centers = sorted(centers, key=lambda x: x[1], reverse=True)[:numeral]

    # 可视化label区域和每个区域内的最大attention点
    plt.figure(figsize=(3, 3))
    plt.imshow(labeled_map, cmap="coolwarm")

    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_labeled_map.png"), bbox_inches="tight"
    )
    for center, _ in centers:
        y, x = center[1], center[0]  # 注意顺序 (列, 行)，对应 matplotlib 的 (x, y)
        plt.scatter(y, x, c="red", s=20, marker="x")
    for center, _ in top_centers:
        y, x = center[1], center[0]  # 注意顺序 (列, 行)，对应 matplotlib 的 (x, y)
        plt.scatter(y, x, c="blue", s=20, marker="x")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_labeled_scattered_map.png"),
        bbox_inches="tight",
    )
    plt.close()

    # 创建 mask
    mask = np.ones_like(attention_map_np, dtype=np.float32)
    for center, _ in top_centers:
        x, y = int(center[0]), int(center[1])
        cv2.circle(
            mask, (y, x), radius=8, color=strength, thickness=-1
        )  # 增强一个小圆区域

    os.makedirs(save_dir, exist_ok=True)
    # 可视化 mask
    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="coolwarm")
    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{prefix}_mask.png"), bbox_inches="tight")
    plt.close()

    # 也可视化原始 attention_map
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap="viridis")
    plt.title("Original Attention Map")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_attention_map.png"), bbox_inches="tight"
    )
    plt.close()

    # 应用 mask 到 latents
    mask = torch.from_numpy(mask).to(latents.device, dtype=latents.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 匹配 latents 的维度
    latents = latents * mask

    return latents


def noise_recreate_4(
    latents,
    attention_map,
    numeral,
    alpha=1.1,
    sigma=8,
    min_distance=15,
    threshold_rel=0.25,
    save_dir="./vis_output",
    prefix="sample",
):
    """
    基于cross-attention map的空间分布特征创建mask，增强指定区域的噪声水平
    Args:
        latents: 潜在变量 [batch, channels, height, width]
        attention_map: 注意力热力图 [height, width]
        numeral: 需要增强的区域数量
        alpha: 增强强度系数
        sigma: 高斯模糊半径(控制区域大小)
        min_distance: 峰值点最小间距
        threshold_rel: 峰值检测的相对阈值(0-1)
        save_dir: 可视化保存路径
        prefix: 文件前缀
    Returns:
        latents: 增强后的潜在变量
    """

    # -------------------------
    # 准备attention map数据
    # -------------------------
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    attention_map_np = (
        attention_map.cpu().numpy()
        if isinstance(attention_map, torch.Tensor)
        else attention_map
    )
    # -------------------------
    # 创建mask
    # -------------------------
    centers = peak_local_max(
        attention_map_np,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=numeral,
    )
    mask = np.ones_like(attention_map_np, dtype=np.float32)
    for center in centers:
        x, y = int(center[0]), int(center[1])
        cv2.circle(
            mask, (y, x), radius=sigma, color=alpha, thickness=-1
        )  # 增强一个小圆区域

    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="coolwarm")
    for center in centers:
        y, x = center[1], center[0]  # 注意顺序 (列, 行)，对应 matplotlib 的 (x, y)
        plt.scatter(y, x, c="red", s=20, marker="x")  # 用黑色叉号标注中心点

    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_labeled_map.png"), bbox_inches="tight"
    )
    plt.close()

    os.makedirs(save_dir, exist_ok=True)
    # 可视化 mask
    plt.figure(figsize=(3, 3))
    plt.imshow(mask, cmap="coolwarm")
    plt.title("Mask")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{prefix}_mask.png"), bbox_inches="tight")
    plt.close()

    # 也可视化原始 attention_map
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap="viridis")
    plt.title("Original Attention Map")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(
        os.path.join(save_dir, f"{prefix}_attention_map.png"), bbox_inches="tight"
    )
    plt.close()

    # 应用 mask 到 latents
    mask = torch.from_numpy(mask).to(latents.device, dtype=latents.dtype)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 匹配 latents 的维度
    latents = latents * mask

    return latents


def noise_adjust_patch(latents, attention_map, numeral):
    """
    根据attention map的高响应部分来重组latents
    目标：使noise在去噪时产生numeral个独立的attention高响应区域
    方式：
        1. 找出attention map中numeral个最高响应区域
        2. 计算这些区域的中心坐标
        3. 在这些区域周围增强noise强度
        4. 在其他区域减弱noise强度
    """

    # 将latents分成3x3的patch
    patch_size = 128 // 3

    # 找出attention map中numeral个最高响应patch
    patch_scores = []
    for i in range(3):
        for j in range(3):
            h_start = i * patch_size
            h_end = (i + 1) * patch_size if i < 2 else 128
            w_start = j * patch_size
            w_end = (j + 1) * patch_size if j < 2 else 128

            # 计算每个patch的平均attention值
            patch_attention = attention_map[h_start:h_end, w_start:w_end]
            patch_score = patch_attention.mean()
            patch_scores.append((i, j, patch_score))

    # 按attention值排序并选择前numeral个patch
    patch_scores.sort(key=lambda x: -x[2])
    top_patches = patch_scores[:numeral]

    # 增强选定patch的noise强度
    for i, j, _ in top_patches:
        h_start = i * patch_size
        h_end = (i + 1) * patch_size if i < 2 else 128
        w_start = j * patch_size
        w_end = (j + 1) * patch_size if j < 2 else 128

        # 增加选定patch的noise强度
        latents[:, :, h_start:h_end, w_start:w_end] *= 1.1

    # 减弱其他patch的noise强度
    for i, j, _ in patch_scores[numeral:]:
        h_start = i * patch_size
        h_end = (i + 1) * patch_size if i < 2 else 128
        w_start = j * patch_size
        w_end = (j + 1) * patch_size if j < 2 else 128

        # 减少其他patch的noise强度
        latents[:, :, h_start:h_end, w_start:w_end] *= 0.8

    return latents


def latents_compute_mean_std(latents):
    patch_size = 128 // 3
    mean = latents.mean().item()
    std = latents.std().item()
    print(f"latents mean: {mean:.4f}, std: {std:.4f}")
    for i in range(3):
        for j in range(3):
            h_start = i * patch_size
            h_end = (i + 1) * patch_size if i < 2 else 128
            w_start = j * patch_size
            w_end = (j + 1) * patch_size if j < 2 else 128
            patch = latents[:, :, h_start:h_end, w_start:w_end]
            mean = patch.mean().item()
            std = patch.std().item()
            print(f"patch ({i},{j}) mean: {mean:.4f}, std: {std:.4f}")


def perform_gen(pipe, seed, prompt, step, attention_map, numeral):
    controller = AttentionStore()
    ptp_utils.register_attention_control(model=pipe, controller=controller)
    g = torch.Generator(DEVICE).manual_seed(seed)
    latents = torch.randn(
        (1, 4, 128, 128), generator=g, device=DEVICE, dtype=torch.float16
    )
    # latents_compute_mean_std(latents)

    # latents = noise_adjust_patch(latents, attention_map, numeral)
    alpha_content = 1.07
    alpha_edge = 0.9
    alpha_background = 0.95
    output_path = f"/data/wsq_data/count_result/attention_map_vis_noise_create_mask_v2_3_{alpha_content}_{alpha_edge}_{alpha_background}/{obj}/{num}/"
    latents = noise_recreate_2_3(
        latents,
        attention_map,
        numeral,
        alpha_content=alpha_content,
        alpha_edge=alpha_edge,
        alpha_background=alpha_background,
        save_dir=output_path,
        prefix=str(seed),
    )
    # latents = noise_recreate_2_2(latents, attention_map, numeral, prefix=str(seed))

    # latents_compute_mean_std(latents)

    # """

    image = pipe(prompt=prompt, num_inference_steps=step, latents=latents).images[0]
    # image.save(f"{prompt}_{seed}.png")
    place_in_unet_set = set()
    place_in_unet_set.add("down_blocks.2.attentions.1")
    # for name in pipe.unet.attn_processors.keys():
    #     place_in_unet = ".".join(name.split(".")[:-4])
    #     place_in_unet_set.add(place_in_unet)
    for place_in_unet in list(place_in_unet_set):
        pil_img = vis_utils.show_cross_attention(
            attention_store=controller,
            prompt=prompt,
            tokenizer=pipe.tokenizer,
            res=32,
            from_where=(place_in_unet,),
            indices_to_alter=[5],
            orig_image=image,
        )
        if pil_img == False:
            print(f"{place_in_unet} is NULL")
            continue
        output_path = f"/data/wsq_data/count_result/attention_map_vis_noise_create_v2_3_{alpha_content}_{alpha_edge}_{alpha_background}/{obj}/{num}/"
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, f"seed{seed}_step{step}_{prompt}.png")
        pil_img.save(path)
        print(f"result attention map save to {path}")

    # """


if __name__ == "__main__":

    model_id = PIPELINE_PATH
    pipe = AttentionStableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe = pipe.to(DEVICE)

    seeds = [43, 73, 1234]
    prompts, object_list, num_int_list = create_prompt_list()
    # prompts = [
    #     "a photo of three apples",
    # ]
    # object_list = [
    #     "apple",
    # ]
    # num_int_list = [3]
    steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50]
    for p_id, prompt in enumerate(prompts):
        obj = object_list[p_id]
        num = num_int_list[p_id]
        for seed in seeds:
            # if not (obj == "backpack" and num == 4 and seed == 43):
            #     continue
            print(f"num:{num}, obj : {obj}")
            map_path = f"/data/wsq_data/count_result/attention_map_store/{obj}/{num}/seed{seed}_step10_{prompt}.npy"
            attention_map = vis_utils.read_attention(map_path)
            for step in steps:
                perform_gen(
                    pipe=pipe,
                    seed=seed,
                    prompt=prompt,
                    step=step,
                    attention_map=attention_map,
                    numeral=num,
                )
