from diffusers import DiffusionPipeline
from utils.json_data_operator import create_data_json, read_all_skills
import torch
import os
import tqdm
import random
import random

PIPELINE_PATH = "/data/wsq_data/stable-diffusion-xl-base-1/"
IMG_OUTPUT_PATH = "/data/wsq_data/sd_output/sdxl/images_data"
IND_OUTPUT_PATH = "/data/wsq_data/sd_output/sdxl/prompts_images_indexes"
PROMPT_INPUT_PATH = "data/prompts_data"

DEVICE = "cuda:0"
NUM_INFERENCE_STEPS = 50
NUM_IMAGES_PER_PROMPT = 10
random.seed(42)
SEEDS = [random.randint(0, 2**32 - 1) for _ in range(NUM_IMAGES_PER_PROMPT)]
all_skill = [
    "attribute_binding_color",
    "attribute_binding_material",
    "attribute_binding_pattern",
    "attribute_binding_sentiment",
    "attribute_binding_shape",
    "attribute_comparison",
    "object_counting",
    "object_differentiation",
    "object_negation",
    "object_universality",
    "relationship_interaction",
    "relationship_part",
    "relationship_spatial",
    "cross_attribute_binding_interaction",
    "cross_attribute_binding_part",
    "cross_attribute_binding_part_counting",
    "cross_attribute_binding_spatial",
    "cross_attribute_comparison_attribute_spatial",
    "cross_counting_attribute_binding",
    "cross_counting_attribute_binding_spatial",
    "cross_counting_spatial",
]

# SKILL_NAMES = ['relationship_interaction', 'cross_counting_attribute_binding', 'cross_attribute_comparison_attribute_spatial',  'cross_attribute_binding_spatial']
SKILL_NAMES = ["attribute_binding_material", "object_counting"]


def get_inputs(prompt, batch_size=1, seeds=[]):
    generator = [
        torch.Generator(DEVICE).manual_seed(seeds[i]) for i in range(batch_size)
    ]
    prompts = batch_size * [prompt]
    num_inference_steps = NUM_INFERENCE_STEPS
    return {
        "prompt": prompts,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
    }


if __name__ == "__main__":

    model_id = PIPELINE_PATH
    pipe = DiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe = pipe.to(DEVICE)

    print(
        f"seeds: {SEEDS}, NUM_INFERENCE_STEPS: {NUM_INFERENCE_STEPS}, NUM_IMAGES_PER_PROMPT: {NUM_IMAGES_PER_PROMPT}"
    )
    sum_sum = 0
    for SKILL_NAME in SKILL_NAMES:
        results = read_all_skills(f"{PROMPT_INPUT_PATH}/{SKILL_NAME}")
        sum_value = 0
        # 默认情况下，sorted() 会按键（即字典的键）进行升序排序。
        for level, value in sorted(results.items()):
            for p_id, item in enumerate(value):
                prompt = item["sentence"]
                if prompt != "":
                    sum_value = sum_value + 1
        sum_sum = sum_sum + sum_value
        print(f"skill: {SKILL_NAME}, sum_value: {sum_value}")
    print(f"Total = {sum_sum}")

    for SKILL_NAME in SKILL_NAMES:
        results = read_all_skills(f"{PROMPT_INPUT_PATH}/{SKILL_NAME}")
        # sum_value = 0
        for level, value in sorted(
            results.items()
        ):  # 默认情况下，sorted() 会按键（即字典的键）进行升序排序。
            # sum_value += len(value)
            print(f"skill: {SKILL_NAME}, level: {level}, count: {len(value)}")
            os.makedirs(f"{IMG_OUTPUT_PATH}/{SKILL_NAME}/level_{level}", exist_ok=True)
            os.makedirs(f"{IND_OUTPUT_PATH}/{SKILL_NAME}", exist_ok=True)

            for p_id, item in enumerate(value):
                print(
                    f"------------ skill name: {SKILL_NAME} | level: {level} | p_id: {p_id} | sentence: {item['sentence']} ------------"
                )
                if SKILL_NAME in ["attribute_binding_material"]:
                    if level in ["1", "2", "3", "4"]:
                        continue
                    elif level in ["5"]:
                        if p_id <= 162:
                            continue

                elif SKILL_NAME in ["object_counting"]:
                    if level in ["1", "2", "3", "4", "5"]:
                        continue
                    elif level in ["6"]:
                        if p_id <= 275:
                            continue
                prompt = item["sentence"]
                item["seeds"] = SEEDS
                item["img_path"] = []
                for i_id, seed in enumerate(SEEDS):
                    print(
                        f"------------ skill name: {SKILL_NAME} | level: {level} | p_id: {p_id} | sentence: {item['sentence']} ------------"
                    )
                    print(f"i_id: {i_id}, seed: {seed}")

                    g = torch.Generator(DEVICE).manual_seed(seed)

                    image = pipe(prompt=prompt, generator=g).images[0]

                    IMG_SAVE_PATH = f"{IMG_OUTPUT_PATH}/{SKILL_NAME}/level_{level}/{p_id}_{i_id}_{prompt[:240]}.png"
                    item["img_path"].append(IMG_SAVE_PATH)
                    image.save(IMG_SAVE_PATH)
                create_data_json(
                    f"{IND_OUTPUT_PATH}/{SKILL_NAME}/level_{level}.txt", json_data=item
                )
        # print(f"skill: {SKILL_NAME}, sum_value: {sum_value}")
