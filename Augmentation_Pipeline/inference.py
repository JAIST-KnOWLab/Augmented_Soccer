import torch
from MMTBART_model import Frame_Predict_Event_Model
from dataset_inf import Short_Term_Dataset
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "MODEL_PATH"
feature_root_path = "FEATURE_ROOT_PATH"

"""
i3d_2fps 1024
c3d_2fps 4096
baidu  8576
resnet_2fps_512  512
resnet_5fps  2048
clip 512
"""
ann_root_path = "TEXT_PATH"

model = Frame_Predict_Event_Model(inference=True, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

json_files = [os.path.join(root, file) for root, _, files in os.walk(ann_root_path) for file in files if file.endswith(".json")]

for file_path in tqdm(json_files, desc="Processing JSON files"):
    print(f"\nProcessing: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data or len(data["annotations"]) == 0:
        print(f"{file_path} do not have `annotations` or it is empty, skipping.")
        continue

    dataset = Short_Term_Dataset(feature_root_path, file_path)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collater)

    predictions = []

    for sample in tqdm(dataloader, desc=f"Processing {file_path}", leave=True, position=0):
        sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}  
        output = model(sample)

        caption_info = sample["caption_info"][0]  
        description = caption_info[-2] 
        game_time = caption_info[1]  
        ori_text = caption_info[-1]

        if isinstance(output, list) and len(output) > 0:
            predictions.append({
                "game_time": game_time,
                "description": description,
                "ori_text": ori_text,
                "short-term": output[0]
            })

    feature_tag = os.path.basename(feature_root_path.rstrip("/"))
    output_file_path = file_path.replace(".json", f"_{feature_tag}.json")

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"\n Saved to: {output_file_path}")


