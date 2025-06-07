from utils.attention_sdxl_pipeline import AttentionStableDiffusionXLPipeline
import os
import torch
import random
import numpy as np
from typing import List, Union
import cv2
from PIL import Image

PIPELINE_PATH = "/data/wsq_data/stable-diffusion-xl-base-1/"
DEVICE = "cuda:0"


def set_seed(seed: int) -> torch.Generator:
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return gen


def resample_noise(noise: torch.Tensor, noise_indexes: List[int]):
    """
    Resample noise based on the given noise index.

    Args:
        noise (torch.Tensor): The input noise tensor.
        noise_indexes (List[int]): The index to resample noise.

    Returns:
        torch.Tensor: The resampled noise tensor.
    """
    # 将noise划分为4x4个patch，每个patch为32x32
    assert noise.shape == (1, 4, 128, 128), "noise shape must be [1, 4, 128, 128]"
    patch_size = 32
    num_patches = 4
    noise_clone = noise.clone()
    for noise_index in noise_indexes:
        # 计算patch的行列索引
        row = noise_index // num_patches
        col = noise_index % num_patches
        # 替换指定patch
        noise_clone[
            :,
            :,
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ] = torch.randn(
            (1, 4, patch_size, patch_size), device=noise.device, dtype=noise.dtype
        )
    return noise_clone


def insert_noise(noise1: torch.Tensor, noise2: torch.Tensor, noise_indexes: List[int]):
    """
    Insert noise2 into noise1 based on the given noise index.

    Args:
        noise1 (torch.Tensor): The input noise tensor.
        noise2 (torch.Tensor): The noise tensor to be inserted.
        noise_indexes (List[int]): The index to insert noise.

    Returns:
        torch.Tensor: The modified noise tensor.
    """
    # 将noise划分为4x4个patch，每个patch为32x32
    assert noise1.shape == (1, 4, 128, 128), "noise shape must be [1, 4, 128, 128]"
    patch_size = 32
    num_patches = 4
    noise_clone = noise1.clone()
    for noise_index in noise_indexes:
        # 计算patch的行列索引
        row = noise_index // num_patches
        col = noise_index % num_patches

        # 替换指定patch
        noise_clone[
            :,
            :,
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ] = noise2[
            :,
            :,
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ]

    # # 创建一个可视化图像来展示修改的patch位置
    # vis_img = np.ones((128, 128, 3), dtype=np.uint8) * 255  # 白色背景
    # for noise_index in noise_indexes:
    #     # 计算patch的行列索引
    #     row = noise_index // num_patches
    #     col = noise_index % num_patches
    #     # 在可视化图像中将修改的patch区域标记为绿色
    #     vis_img[
    #         row * patch_size : (row + 1) * patch_size,
    #         col * patch_size : (col + 1) * patch_size,
    #     ] = [
    #         0,
    #         255,
    #         0,
    #     ]  # 绿色
    # cv2.imwrite(
    #     os.path.join(output_dir, f"noise_patch_visualization_{noise_indexes}.png"),
    #     cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
    # )

    return noise_clone


def insert_noise_add(noise: torch.Tensor, strength: float, noise_indexes: List[int]):
    """
    在指定patch上对noise增加strength，增加后不超过1。

    Args:
        noise (torch.Tensor): 输入的噪声张量。
        strength (float): 增加的强度，0-1之间。
        noise_indexes (List[int]): 需要增加的patch索引列表。

    Returns:
        torch.Tensor: 修改后的噪声张量。
    """
    assert noise.shape == (1, 4, 128, 128), "noise shape must be [1, 4, 128, 128]"
    patch_size = 32
    num_patches = 4
    noise_clone = noise.clone()
    for noise_index in noise_indexes:
        row = noise_index // num_patches
        col = noise_index % num_patches
        patch = noise_clone[
            :,
            :,
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ]
        patch = patch + strength
        patch = patch.clamp(max=1.0)
        noise_clone[
            :,
            :,
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ] = patch
    return noise_clone


def interpolate_noise(noise1: torch.Tensor, noise2: torch.Tensor, alpha: float):
    """
    Interpolate between two noise tensors.

    Args:
        noise1 (torch.Tensor): The first noise tensor.
        noise2 (torch.Tensor): The second noise tensor.
        alpha (float): The interpolation factor.

    Returns:
        torch.Tensor: The interpolated noise tensor.
    """
    return noise1 * alpha + noise2 * (1 - alpha)


if __name__ == "__main__":
    # ==================================================
    # 1. init model pipeline
    # ==================================================
    model_id = PIPELINE_PATH
    pipe = AttentionStableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe = pipe.to(DEVICE)

    # ==================================================
    # 2. create output dir
    # ==================================================

    output_dir = "noise_test_outputs/noise_resample"
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================
    # 3. prepare initial noise
    # ==================================================
    seed1 = 271828
    seed2 = 42
    latents1 = torch.randn(
        (1, 4, 128, 128), generator=set_seed(seed1), device=DEVICE, dtype=torch.float16
    )
    latents2 = torch.randn(
        (1, 4, 128, 128), generator=set_seed(seed2), device=DEVICE, dtype=torch.float16
    )
    # noise_indexes = [12, 13, 14]
    # latents = resample_noise(latents, [4, 5, 6, 7])
    # latents = insert_noise(latents1, latents2, noise_indexes)
    # latents = latents1

    # ==================================================
    #  4. call model
    # ==================================================

    # Original Generation
    # numerals = ["one", "two", "three", "four", "five", "six"]
    # for i, num in enumerate(numerals):
    #     prompt = (
    #         f"a photo of {num} giraffe"
    #         if num == "one"
    #         else f"a photo of {num} giraffes"
    #     )
    #     for seed in [seed1, seed2]:
    #         latents = latents1 if seed == seed1 else latents2
    #         image = pipe(
    #             prompt,
    #             num_inference_steps=50,
    #             latents=latents,
    #         ).images[0]

    #         os.makedirs(os.path.join(output_dir, "original", "giraffe"), exist_ok=True)
    #         image.save(
    #             os.path.join(
    #                 output_dir,
    #                 "original",
    #                 "giraffe",
    #                 f"{i+1}_{seed}_{prompt}.png",
    #             )
    #         )

    numerals = ["three"]
    ilist = [3]
    animal = "horse"
    for i, num in zip(ilist, numerals):
        prompt = (
            f"a photo of {num} {animal}"
            if num == "one"
            else f"a photo of {num} {animal}s"
        )
        for noise_indexes in [[i] for i in range(16)]:
            latents = insert_noise_add(
                latents1, strength=0.3, noise_indexes=noise_indexes
            )
            image = pipe(
                prompt,
                num_inference_steps=50,
                latents=latents,
            ).images[0]

            os.makedirs(os.path.join(output_dir, "add_0.3", animal), exist_ok=True)

            # # 创建一个和生成图像一样大小的半透明绿色遮罩
            # img_array = np.array(image)
            # overlay = np.zeros_like(img_array)
            # for noise_index in noise_indexes:

            #     # 计算noise patch和实际图像的缩放比例
            #     patch_size = 32
            #     num_patches = 4
            #     img_height, img_width = img_array.shape[:2]
            #     scale_h = img_height / (patch_size * num_patches)
            #     scale_w = img_width / (patch_size * num_patches)

            #     # 计算patch在实际图像中的位置
            #     row = noise_index // num_patches
            #     col = noise_index % num_patches
            #     start_h = int(row * patch_size * scale_h)
            #     end_h = int((row + 1) * patch_size * scale_h)
            #     start_w = int(col * patch_size * scale_w)
            #     end_w = int((col + 1) * patch_size * scale_w)

            #     # 在对应位置添加半透明绿色遮罩
            #     overlay[start_h:end_h, start_w:end_w] = [0, 255, 0]
            # alpha = 0.3  # 设置透明度
            # img_array = cv2.addWeighted(img_array, 1, overlay, alpha, 0)
            # image = Image.fromarray(img_array)

            image.save(
                os.path.join(
                    output_dir,
                    "add_0.3",
                    animal,
                    f"{i}_{seed1}_{seed2}_{noise_indexes}_{prompt}.png",
                )
            )
