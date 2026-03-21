import os
import json
import numpy as np
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
import random

random.seed(42)
np.random.seed(42)


ann_root_path = 'TEXT_PATH'

# =========================
# Visual Feature types 
# =========================
model_types = [
  'Visual Feature types here'
]


model_data = {
    model: {
        "gt": [],
        "description": [],
        "short-term": []
    }
    for model in model_types
}


json_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(ann_root_path)
    for f in files if f.endswith(".json")
]

print(f"Found {len(json_files)} JSON files")


for file_path in tqdm(json_files, desc="Processing JSON"):
    fname = os.path.basename(file_path)


    matched_model = None
    for mt in model_types:
        if mt in fname:
            matched_model = mt
            break

    if matched_model is None:
        continue  

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    if isinstance(data, dict):
        data = [data]

    for entry in data:
        if (
            "ori_text" not in entry or
            "description" not in entry or
            "short-term" not in entry
        ):
            continue

        gt = entry["ori_text"].strip()
        des = entry["description"].strip()
        st  = entry["short-term"].strip()

        model_data[matched_model]["gt"].append(gt)
        model_data[matched_model]["description"].append(des)
        model_data[matched_model]["short-term"].append(st)


def compute_cider(pred_list, gt_list):
    cider = Cider()
    pred_dict = {str(i): [p] for i, p in enumerate(pred_list)}
    gt_dict   = {str(i): [g] for i, g in enumerate(gt_list)}
    score, _ = cider.compute_score(gt_dict, pred_dict)
    return score



final_results = {}

for mt in model_types:
    gts = model_data[mt]["gt"]

    if len(gts) == 0:
        final_results[mt] = {
            "CIDEr_description": 0,
            "CIDEr_short-term": 0
        }
        continue

    cider_des = compute_cider(model_data[mt]["description"], gts)
    cider_st  = compute_cider(model_data[mt]["short-term"], gts)

    final_results[mt] = {
        "CIDEr_description": round(cider_des, 3),
        "CIDEr_short-term": round(cider_st, 3),
    }


output_file = "cider_results_by_model.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4, ensure_ascii=False)

print("\n Cider results saved to: ", output_file)
