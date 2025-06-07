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
    OUTPUT_DIR  = "/data/wsq_data/count_result/geckonum_outputs/sd21"
    SEEDS = [42, 15236, 8668521, 4020, 471159]
    PROMPT_TYPE_FOR_RUN = ["numeric_simple", "attribute-color", "numeric_sentence","2-additive","2-additive-color","3-additive"]
    ############===== read dataset ======###########
    dataset = read_csv_to_nested_dict()
    
    for prompt_type in dataset:
        if prompt_type not in PROMPT_TYPE_FOR_RUN:
            continue
        for p_id in dataset[prompt_type]:
            print(f"p_id = {p_id}")
            data = dataset[prompt_type][p_id]
            prompt = data["prompt"]
            has_numeral = data["has_numeral"]
            if has_numeral==1:  #  只跑number形式的number
                continue
            for seed in SEEDS:
                image = generate_image_from_prompt(prompt =prompt, seed = seed)
                img_output_path = f"{OUTPUT_DIR}/{p_id}_{seed}.png"
                print(f"saving to {img_output_path}")
                image.save(img_output_path)

