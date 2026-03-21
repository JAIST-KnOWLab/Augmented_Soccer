import os
import random
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from transformers import BartTokenizer
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100

def parse_labels_caption(half, file_path, league, game, timestamp_key):
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    result = []
    for annotation in data.get('annotations', []):
        game_time = annotation.get(timestamp_key, '')
        if not game_time:
            print(f" {file_path}'s `game_time` is empty.")
            continue

        minutes, seconds = map(int, game_time.split(':'))
        timestamp = minutes * 60 + seconds  

        label = annotation.get('label', '')  
        description = annotation.get('query', '')
        ori_text = annotation.get('short-term', '')
        if not description:
            print(f" {file_path}'s `description` is empty.")
            continue

        result.append((half, timestamp, label, league, game, description, ori_text))

    
    print(f" Completed: {len(result)} data")
    return result


def traverse_and_parse(file_path, timestamp_key="game_time"):
    all_data = []
    if os.path.exists(file_path):
        league = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  #  league 
        game = os.path.basename(os.path.dirname(file_path))  #  game 
        half = 1 if "1_game.json" in file_path else 2
        all_data.extend(parse_labels_caption(half, file_path, league, game, timestamp_key))
    
    return all_data

def load_features(feature_path, timestamp, window, fps=5):    
    if not os.path.exists(feature_path):
        return None
    features = np.load(feature_path)
    total_frames = int(window * 2 * fps)
    if timestamp * fps > len(features):
        return None

    start_frame = int(max(0, timestamp - window) * fps + 1)
    end_frame = int((timestamp + window) * fps + 1)

    if end_frame > len(features):
        start_frame = int(max(0, len(features) - total_frames))

    corres_feature = features[start_frame:start_frame + total_frames]

    return corres_feature

class Short_Term_Dataset(Dataset):
    def __init__(self, 
                 feature_root_path, 
                 file_path,  
                 window=15, 
                 fps=5, 
                 timestamp_key="game_time",
                 tokenizer_name='facebook/bart-large', 
                 max_token_length=256):
        super().__init__()
        
        self.caption = traverse_and_parse(file_path, timestamp_key)
        
        self.feature_root_path = feature_root_path
        self.window = window
        self.fps = fps
        # init tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        special_tokens = ["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])", "[STADIUM]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.max_token_length = max_token_length

    def __getitem__(self, index):
        corres_feature = None
        num_retries = 50

        for _ in range(num_retries):
            try:
                half, timestamp, label, league, game, description, ori_text = self.caption[index]

                # load .npy
                feature_folder = os.path.join(self.feature_root_path, league, game)
                file_paths = [os.path.join(feature_folder, file) for file in os.listdir(feature_folder) 
                              if file.startswith(str(half)) and file.endswith(".npy")]

                corres_feature = torch.from_numpy(load_features(file_paths[0], timestamp, self.window, self.fps))
             

                description_tokens = self.tokenizer(
                    description,
                    return_tensors="pt",
                    max_length=self.max_token_length,
                    truncation=True
                ).input_ids[0]

            except Exception as e:
                print(f"error: {e}")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return {
            "features": corres_feature,
            "input_ids": description_tokens,  
            "caption_info": self.caption[index],
            "event": label
        }
    
    def __len__(self):
        return len(self.caption)
    
    def collater(self, instances):
        events = [instance["event"] for instance in instances]
        input_ids = [instance["input_ids"] for instance in instances]

        max_length = max(len(x) for x in input_ids)

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        batch = {
            "input_ids": input_ids,
            "caption_info": [instance["caption_info"] for instance in instances],
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "events": events
        }

        if "features" in instances[0]:
            features = [instance["features"] for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch["features"] = torch.stack(features)
            else:
                batch["features"] = features

        return batch
