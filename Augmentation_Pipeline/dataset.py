import os
import random
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from transformers import BartTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
IGNORE_INDEX = -100

def parse_labels_caption(half, file_path, league, game, timestamp_key):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    result = []
    for annotation in data.get('annotations', []):
        try:
            game_time = annotation.get(timestamp_key, '')
            if isinstance(game_time, int):  
                game_time = f"{game_time // 60}:{game_time % 60:02d}" 
            elif isinstance(game_time, str):
                if ':' not in game_time:  
                    raise ValueError(f"Invalid game_time format: {game_time}")
            else:
                raise TypeError(f"Unexpected game_time type: {type(game_time)}")

            minutes, seconds = map(int, game_time.split(':'))

            timestamp = minutes * 60 + seconds  
            
            label = annotation.get('label', '')
            sncaption_anonymized = annotation.get('query')
            short_term_anonymized = annotation.get('short-term')
            
            result.append((half, timestamp, label, league, game, sncaption_anonymized, short_term_anonymized))
        
        except ValueError as e:
            print(f"Error processing annotation: {annotation}")
            print(f"Error message: {str(e)}")
            continue
    
    return result

def traverse_and_parse(root_dir, timestamp_key="game_time"):
    all_data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file in ['1_game.json', '2_game.json']:
                league = os.path.basename(os.path.dirname(subdir))
                game = os.path.basename(subdir)
                file_path = os.path.join(subdir, file)
                half = 1 if file == '1_game.json' else 2
                all_data.extend(parse_labels_caption(half, file_path, league, game, timestamp_key))
    
    return all_data

def load_features(feature_path, timestamp, window, fps=5):
    features = np.load(feature_path)
    total_frames = int(window * 2 * fps) 
    if timestamp * fps > len(features):
        return None
    
    start_frame = int(max(0, timestamp - window) * fps + 1)
    end_frame = int((timestamp + window) * fps + 1)
    if end_frame > len(features):
        start_frame = int(max(0, len(features) - total_frames))  
    corres_feature = features[start_frame:start_frame+total_frames]
    return corres_feature

class Short_Term_Dataset(Dataset):
    def __init__(self, 
                 feature_root_path, 
                 ann_root_path, 
                 window=15, 
                 fps=5, 
                 timestamp_key="game_time",
                 tokenizer_name ='facebook/bart-large', 
                 max_token_length=256
                 ):
        super().__init__()
        self.caption = traverse_and_parse(ann_root_path, timestamp_key)
        if not self.caption:
            raise ValueError(f"No annotations found in {ann_root_path}. Please check the directory and timestamp key.")
        self.feature_root_path = feature_root_path
        self.window = window
        self.fps = fps
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        special_tokens = ["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])", "[STADIUM]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.max_token_length = max_token_length

    def __getitem__(self, index):
        corres_feature = None
        num_retries = 50
        for _ in range(num_retries):
            try:
                half, timestamp, label, league, game, sncaption_anonymized, short_term_anonymized = self.caption[index]
                feature_folder = os.path.join(self.feature_root_path, league, game)
                file_paths = [os.path.join(feature_folder, file) for file in os.listdir(feature_folder) if file.startswith(str(half)) and file.endswith(".npy")]
                corres_feature = torch.from_numpy(load_features(file_paths[0], timestamp, self.window, self.fps))
                if corres_feature is None:
                    raise ValueError(f"Failed to load valid features from {file_paths[0]} at timestamp {timestamp}")

                sncaption_anonymized_tokens = self.tokenizer(
                            sncaption_anonymized,
                            return_tensors="pt",
                            max_length=self.max_token_length,
                            truncation=True
                    ).input_ids[0]
                
                short_term_anonymized_tokens = self.tokenizer(
                            short_term_anonymized,
                            return_tensors="pt",
                            max_length=self.max_token_length,
                            truncation=True
                    ).input_ids[0]
            except:
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return {
            "features": corres_feature,
            "sncaption_anonymized_ids": sncaption_anonymized_tokens,
            "short_term_anonymized_ids": short_term_anonymized_tokens,
            "caption_info": self.caption[index],
            "event": label
        }
    
    def __len__(self):
        return len(self.caption)
    
    def collater(self, instances):
        events = [instance["event"] for instance in instances]

        sncaption_anonymized_ids = [instance["sncaption_anonymized_ids"]  for instance in instances]
        short_term_anonymized_ids = [instance["short_term_anonymized_ids"] for instance in instances]

        max_length = max(max(len(x) for x in sncaption_anonymized_ids), max(len(x) for x in short_term_anonymized_ids))

        input_ids = pad_sequence(
            sncaption_anonymized_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(
            short_term_anonymized_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
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
