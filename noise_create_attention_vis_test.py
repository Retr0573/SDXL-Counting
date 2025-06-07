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
import matplotlib.colors as mcolors
from skimage.morphology import reconstruction
from skimage.draw import disk

p = inflect.engine()

PIPELINE_PATH = "/data/wsq_data/stable-diffusion-xl-base-1/"

DEVICE = "cuda:1"


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


def noise_recreate(
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
    """ """

    def process_attention_map(att_map, steps=1):
        """
        norm and square
        """
        for _ in range(steps):
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
            att_map = att_map**2
        # 最后再归一化一次
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        return att_map

    def cluster_high_attention_regions(att_map, numeral=5, percentile=70):
        """
        设置threshold先做一遍过滤，然后使用 KMeans 聚类找到最显著的 numeral 个区域（cluster_masks） 和centers

        """
        h, w = att_map.shape

        # 1. 扁平化和计算阈值（只保留高 attention 区域）
        values = att_map.flatten()
        threshold = 0.1  # 可以改为 np.percentile(values, percentile) 进行自适应阈值
        high_att_mask = values >= threshold
        high_att_mask_2d = high_att_mask.reshape(att_map.shape)

        # 2. 只保留高 attention 区域的坐标和值
        coords = np.array([(i, j) for i in range(h) for j in range(w)])
        coords = coords[high_att_mask]
        weights = values[high_att_mask]

        # 3. 归一化权重
        weights = weights / (weights.sum() + 1e-8)

        # 4. 聚类（KMeans）
        if len(coords) < numeral:
            # 少于 numeral 个点，就直接返回这些点作为中心
            centers = coords
            labels = np.arange(len(coords))  # 每个点一个 cluster
        else:
            kmeans = KMeans(n_clusters=numeral, random_state=42)
            kmeans.fit(coords, sample_weight=weights)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

        # 5. 返回整型像素坐标
        centers = np.round(centers).astype(int)

        # 6. 为每个 center 构造 mask
        cluster_masks = []
        for i in range(numeral):
            mask = np.zeros((h, w), dtype=bool)
            cluster_coords = coords[labels == i]
            for y, x in cluster_coords:
                mask[y, x] = True
            cluster_masks.append(mask)

        # ------------------------- vis start -------------------------

        os.makedirs(save_dir, exist_ok=True)

        fig, axs = plt.subplots(1, 3, figsize=(15, 7))  # 1 row, 3 columns
        axs = axs.flatten()

        # 1. 显示 Normalized Attention Map
        axs[0].imshow(att_map, cmap="viridis")
        axs[0].set_title("Normalized Attention Map")
        axs[0].axis("off")
        axs[0].scatter(
            centers[:, 1], centers[:, 0], c="red", marker="x", s=50, label="Centers"
        )
        axs[0].legend()

        # 2. 显示 Thresholded Mask
        axs[1].imshow(high_att_mask_2d, cmap="gray")
        axs[1].set_title(f"Thresholded Mask (>{threshold})")
        axs[1].axis("off")
        axs[1].scatter(
            centers[:, 1], centers[:, 0], c="red", marker="x", s=50, label="Centers"
        )
        axs[1].legend()

        # 3. Cluster Masks in One Subplot (Color-coded)
        h, w = att_map.shape
        color_mask = np.zeros((h, w, 3), dtype=np.float32)
        cmap_list = list(mcolors.TABLEAU_COLORS.values()) + list(
            mcolors.CSS4_COLORS.values()
        )

        for idx, mask in enumerate(cluster_masks):
            color = mcolors.to_rgb(cmap_list[idx % len(cmap_list)])
            for c in range(3):
                color_mask[..., c] += mask * color[c]  # 不重叠区域就会单独显示颜色

        # clip 到 [0,1]，避免颜色溢出
        color_mask = np.clip(color_mask, 0, 1)
        axs[2].imshow(color_mask)
        axs[2].scatter(
            centers[:, 1], centers[:, 0], c="white", marker="x", s=50, label="Centers"
        )
        axs[2].legend()
        axs[2].set_title("All Cluster Masks (Color Coded)")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_filter_result.png"), dpi=300)
        plt.close()
        # ------------------------- vis end -------------------------

        return centers, high_att_mask_2d, cluster_masks

    def generate_compact_mask(center, cluster_mask, vertical_trim=4, horizontal_trim=4):
        """
        对 cluster_mask 做压缩收缩处理，产生一个更紧凑的 mask：
        1. 水平塌缩：每行的前景像素向中心列靠拢
        2. 垂直塌缩：每列的前景像素向中心行靠拢
        3. 去除上下边界若干行（vertical_trim）
        4. 去除左右边界若干列（horizontal_trim）

        Args:
            center: tuple (cy, cx)，压缩中心位置
            cluster_mask: np.ndarray，原始 bool 或二值 mask
            vertical_trim: int，垂直修剪行数
            horizontal_trim: int，水平修剪列数

        Returns:
            compact_mask: np.ndarray，同尺寸布尔类型的紧凑 mask
        """
        h, w = cluster_mask.shape
        cy, cx = center
        compact_mask = np.zeros_like(cluster_mask, dtype=bool)

        # --- Step 1: 水平塌缩 ---
        for y in range(h):
            row = cluster_mask[y]
            left_count = np.sum(row[:cx])
            right_count = np.sum(row[cx + 1 :])

            # 左侧前景像素向中心靠拢
            if left_count > 0:
                start = max(0, cx - left_count)
                compact_mask[y, start : cx + 1] = True

            # 右侧前景像素向中心靠拢
            if right_count > 0:
                end = min(w, cx + 1 + right_count)
                compact_mask[y, cx + 1 : end] = True

        # --- Step 2: 垂直塌缩 ---
        new_mask = np.zeros_like(compact_mask, dtype=bool)
        for x in range(w):
            col = compact_mask[:, x]
            top_count = np.sum(col[:cy])
            bottom_count = np.sum(col[cy + 1 :])

            # 上方前景像素向中心靠拢
            if top_count > 0:
                start = max(0, cy - top_count)
                new_mask[start : cy + 1, x] = True

            # 下方前景像素向中心靠拢
            if bottom_count > 0:
                end = min(h, cy + 1 + bottom_count)
                new_mask[cy + 1 : end, x] = True

        compact_mask = new_mask

        # --- Step 3: 垂直修剪 ---
        row_indices = np.where(np.any(compact_mask, axis=1))[0]
        if len(row_indices) > 2 * vertical_trim:
            trim_rows = np.concatenate(
                [row_indices[:vertical_trim], row_indices[-vertical_trim:]]
            )
            compact_mask[trim_rows, :] = False
        else:
            compact_mask[:, :] = False

        # --- Step 4: 水平修剪 ---
        col_indices = np.where(np.any(compact_mask, axis=0))[0]
        if len(col_indices) > 2 * horizontal_trim:
            trim_cols = np.concatenate(
                [col_indices[:horizontal_trim], col_indices[-horizontal_trim:]]
            )
            compact_mask[:, trim_cols] = False
        else:
            compact_mask[:, :] = False

        return compact_mask

    processed_map = process_attention_map(attention_map, steps=2)
    os.makedirs(save_dir, exist_ok=True)
    # 聚类，并生成每个cluster的mask
    centers, high_att_mask_2d, cluster_masks = cluster_high_attention_regions(
        processed_map, numeral=numeral
    )
    # 进一步处理每个cluster的mask，令其向center塌缩聚集到
    compact_masks = []
    for idx, mask in enumerate(cluster_masks):
        center = tuple(map(int, centers[idx]))  # 注意 center 是 (y, x)
        compact = generate_compact_mask(center, mask)
        compact_masks.append(compact)

    # ------------------------- vis start -------------------------
    h, w = compact_masks[0].shape
    color_mask = np.zeros((h, w, 3), dtype=np.float32)

    cmap_list = list(mcolors.TABLEAU_COLORS.values()) + list(
        mcolors.CSS4_COLORS.values()
    )
    # 给每个 compact mask 上色
    for idx, mask in enumerate(compact_masks):
        color = mcolors.to_rgb(cmap_list[idx % len(cmap_list)])
        for i in range(3):  # R, G, B 通道
            color_mask[..., i] += mask * color[i]

    # 为防止重叠位置叠加值 > 1，做 clip
    color_mask = np.clip(color_mask, 0, 1)

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(color_mask)
    for center in centers:
        y, x = center
        plt.plot(x, y, "rx", markersize=8, label="Center")
    plt.title("Compact Cluster Masks (Color Coded)")
    plt.savefig(os.path.join(save_dir, f"{prefix}_compact_masks.png"), dpi=300)
    plt.axis("off")
    plt.close()
    # ------------------------- vis end -------------------------

    # 根据compact_masks列表，生成一个新的mask，用于修改latents
    # 假设 compact_masks 是一个布尔数组列表，processed_map 是参考大小的数组
    mask = np.full_like(processed_map, fill_value=alpha_background, dtype=np.float32)
    # 合并所有 compact_mask 为一个总的 bool mask
    combined_mask = np.any(compact_masks, axis=0)  # shape: same as each compact_mask
    # 在对应位置赋值
    mask[combined_mask] = alpha_content

    # 5. 可视化 mask
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
    alpha_content = 1.1
    alpha_edge = 0.9
    alpha_background = 0.95
    output_path = f"/data/wsq_data/count_result/attention_map_vis_noise_create_mask_test_{alpha_content}_{alpha_edge}_{alpha_background}/{obj}/{num}/"
    latents = noise_recreate(
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
        output_path = f"/data/wsq_data/count_result/attention_map_vis_noise_create_test_{alpha_content}_{alpha_edge}_{alpha_background}/{obj}/{num}/"
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, f"seed{seed}_step{step}_{prompt}.png")
        pil_img.save(path)
        print(f"result attention map save to {path}")


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
