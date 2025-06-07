import json
import os
import glob


def create_data_json(
    file_path, objects=(), tags={}, prototype="", sentence="", json_data=None
):

    with open(file_path, "a") as file:
        if json_data is None:
            json_data = {
                "meta_data": {"objects": objects, "tags": tags},
                "prototype": prototype,
                "sentence": sentence,
            }
        json_string = json.dumps(json_data)
        file.write(json_string + "\n")


def read_file(file_path):
    datas = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                # 解析JSON字符串为Python字典
                json_data = json.loads(line.strip())
                datas.append(json_data)

            except json.JSONDecodeError:
                print(f"无法解析行：{line.strip()}")
    return datas


def read_all_skills(aspect_dir):
    results = {}
    txt_file_paths = glob.glob(os.path.join(aspect_dir, "*.txt"))
    for path in txt_file_paths:
        print(path)
    # 依次读取每个 txt 文件
    for txt_file_path in txt_file_paths:
        # 检查文件名里有没有包含数字，将该数字记录成一个变量：level
        level = "".join(
            filter(str.isdigit, os.path.basename(txt_file_path))
        )  # like: "3"
        results.setdefault(level, [])
        is_cross = "_".join(
            os.path.basename(txt_file_path).split("_")[:2]
        )  # like: "in_class"
        files_content_json = read_file(txt_file_path)
        for file_content in files_content_json:
            file_content["meta_data"]["tags"]["level"] = level
            file_content["meta_data"]["tags"]["is_cross"] = is_cross
        results[level].extend(files_content_json)
    return results


if __name__ == "__main__":
    results = read_all_skills("data/prompts_data/attribute_binding_color")
    for level, value in results.items():
        print(f"level: {level}, count: {len(value)}")
