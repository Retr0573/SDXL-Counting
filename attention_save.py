"""
cross attention map 保存
保存某个step的attention map，比如step10，用于后续找高CA响应部分的noise做复制时提供CA参考
"""

from utils.attention_sdxl_pipeline import AttentionStableDiffusionXLPipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import torch
import os
import inflect

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


if __name__ == "__main__":

    model_id = PIPELINE_PATH
    pipe = AttentionStableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    pipe = pipe.to(DEVICE)

    seeds = [43, 73, 1234]
    prompts, object_list, num_int_list = create_prompt_list()
    steps = [10]
    for p_id, prompt in enumerate(prompts):
        obj = object_list[p_id]
        num = num_int_list[p_id]
        for seed in seeds:
            for step in steps:
                controller = AttentionStore()
                ptp_utils.register_attention_control(model=pipe, controller=controller)
                g = torch.Generator(DEVICE).manual_seed(seed)
                image = pipe(
                    prompt=prompt, num_inference_steps=step, generator=g
                ).images[0]
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
                    output_path = (
                        f"/data/wsq_data/count_result/attention_map_store/{obj}/{num}/"
                    )
                    os.makedirs(output_path, exist_ok=True)

                    path = os.path.join(
                        output_path, f"seed{seed}_step{step}_{prompt}.png"
                    )
                    pil_img.save(path)
                    print(f"result attention map save to {path}")

                    path = os.path.join(output_path, f"seed{seed}_step{step}_{prompt}")
                    vis_utils.save_attention(
                        attention_store=controller,
                        prompt=prompt,
                        tokenizer=pipe.tokenizer,
                        res=32,
                        from_where=(place_in_unet,),
                        indices_to_alter=[5],
                        save_path=path,
                    )
                    print(f"result attention map save to {path}")
