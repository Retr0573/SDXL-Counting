from diffusers import DiffusionPipeline
import torch
from PIL import Image
import random
from read_geckonum import read_csv_to_nested_dict
import os
PIPELINE_PATH = "/data/wsq_data/stable-diffusion-2-1-base/"

DEVICE = "cuda:1"
model_id = PIPELINE_PATH
pipe = DiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
pipe = pipe.to(DEVICE)

def generate_image_from_prompt(prompt, seed = 271828):
    """
    根据输入的 prompt 返回生成的图像。
    :param prompt: str, 输入的描述性 prompt
    :return: PIL.Image.Image, 生成的图像
    """
    # 模拟生成图像的逻辑
    g = torch.Generator(DEVICE).manual_seed(seed)

    image = pipe(prompt=prompt,
                 generator=g,
                 guidance_scale = 7.5).images[0]
    return image


if __name__ == "__main__":
    prompt = "1 fish."
    seed =4020
    image = generate_image_from_prompt(prompt =prompt, seed = seed)
    image.save(f"{prompt}_{seed}.png")