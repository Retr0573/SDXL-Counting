import csv
from collections import defaultdict
import os
def read_csv_to_nested_dict(filepath = "counting_matter/geckonum_prompts.csv"):
    print(os.getcwd())
    
    nested_dict = defaultdict(lambda: defaultdict(dict))
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            prompt_type = row['prompt_type']
            index = int(row['index'])
            nested_dict[prompt_type][index] = {
                'prompt': row['prompt'],
                'has_numeral': int(row['has_numeral']),
                'is_frequent': int(row['is_frequent']),
                'entities': row['entities'],
                'dataset_id': row['dataset_id']
            }
    return nested_dict
if __name__ == "__main__":
    PROMPT_TYPE_FOR_RUN = ["numeric_simple", "attribute-color", "numeric_sentence","2-additive","2-additive-color","3-additive"]
    dataset_path = "counting_matter/geckonum_prompts.csv"
    dataset = read_csv_to_nested_dict(dataset_path)
    all_prompt_list = []
    for prompt_type in dataset:
        if prompt_type not in PROMPT_TYPE_FOR_RUN:
            continue
        prompt_list = []
        for p_id in dataset[prompt_type]:
            # print(f"p_id = {p_id}")
            data = dataset[prompt_type][p_id]
            prompt = data["prompt"]
            has_numeral = data["has_numeral"]
            # if has_numeral==1:  #  只跑word形式的number
            #     continue
            prompt_list.append(prompt)
            all_prompt_list.append(prompt)
        print(prompt_type, len(prompt_list))
    print("all_prompt_list", len(all_prompt_list))
    print("all_prompt_list*5", len(all_prompt_list)*5)