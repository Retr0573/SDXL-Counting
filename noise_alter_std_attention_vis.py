"""
noise更改测试

对某个位置的noise进行重采样，将其转化为std更小（比如0.8）的正态分布，
"""

from utils.attention_sdxl_pipeline import AttentionStableDiffusionXLPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import torch
import os
import inflect

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


def adjust_patch_std(latents, patch_index, target_std):
    """
    调整指定patch的标准差

    Args:
        latents: 原始噪声张量 [1,4,128,128]
        patch_index: 要修改的patch索引 (0-8)
        target_std: 目标标准差
    """
    patch_size = 128 // 3
    i = patch_index // 3  # 行索引
    j = patch_index % 3  # 列索引

    # 提取patch
    h_start = i * patch_size
    h_end = (i + 1) * patch_size if i < 2 else 128
    w_start = j * patch_size
    w_end = (j + 1) * patch_size if j < 2 else 128
    patch = latents[:, :, h_start:h_end, w_start:w_end]

    # 计算当前均值和标准差
    current_mean = patch.mean()
    current_std = patch.std()

    # 调整标准差
    adjusted_patch = (patch - current_mean) * (target_std / current_std) + current_mean

    # 替换原patch
    latents[:, :, h_start:h_end, w_start:w_end] = adjusted_patch
    return latents


def latents_operation(latents):
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


def perform_gen(pipe, seed, prompt, step, patch_index=0, target_std=0.8):
    controller = AttentionStore()
    ptp_utils.register_attention_control(model=pipe, controller=controller)
    g = torch.Generator(DEVICE).manual_seed(seed)
    latents = torch.randn(
        (1, 4, 128, 128), generator=g, device=DEVICE, dtype=torch.float16
    )
    latents_operation(latents)

    latents = adjust_patch_std(latents, patch_index=patch_index, target_std=target_std)
    latents_operation(latents)

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
        output_path = f"/data/wsq_data/count_result/attention_map_vis_noise_alter/std_{target_std}/{obj}/{num}/"
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(
            output_path, f"seed{seed}_patch_{patch_index}_step{step}_{prompt}.png"
        )
        pil_img.save(path)
        print(f"result attention map save to {path}")


if __name__ == "__main__":

    model_id = PIPELINE_PATH
    pipe = AttentionStableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe = pipe.to(DEVICE)

    seeds = [73, 43, 271828]
    prompts, object_list, num_int_list = create_prompt_list()
    prompts = [
        "a photo of four bowls",
        "a photo of four donuts",
        "a photo of four cats",
        "a photo of four dogs",
        "a photo of five bowls",
        "a photo of five donuts",
        "a photo of five cats",
        "a photo of five dogs",
    ]
    object_list = [
        "bowl",
        "donut",
        "cat",
        "dog",
        "bowl",
        "donut",
        "cat",
        "dog",
    ]
    num_int_list = [4, 4, 4, 4, 5, 5, 5, 5]
    steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50]
    for p_id, prompt in enumerate(prompts):
        obj = object_list[p_id]
        num = num_int_list[p_id]
        for seed in seeds:
            for step in steps:
                for patch_index in [0, 4, 8]:
                    perform_gen(
                        pipe=pipe,
                        seed=seed,
                        prompt=prompt,
                        step=step,
                        patch_index=patch_index,
                        target_std=0.5,
                    )
