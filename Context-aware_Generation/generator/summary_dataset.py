import os
import json
import glob
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from transformers import BartTokenizer

FEATURE_ROOT = "FEATURE_ROOT_PATH_HERE"  
FEATURE_CONFIG = {
    "c3d_2fps": {"root": os.path.join(FEATURE_ROOT, "features", "c3d_2fps"), "fps": 2, "dim": 4096},
    "i3d_2fps": {"root": os.path.join(FEATURE_ROOT, "features", "i3d_2fps"), "fps": 2, "dim": 1024},
    "resnet_2fps_512": {"root": os.path.join(FEATURE_ROOT, "features", "resnet_2fps_512"), "fps": 2, "dim": 512},
    "resnet_5fps": {"root": os.path.join(FEATURE_ROOT, "features", "resnet_5fps"), "fps": 5, "dim": 2048},
    "clip": {"root": os.path.join(FEATURE_ROOT, "features", "clip"), "fps": 2, "dim": 512},
    "baidu": {"root": os.path.join(FEATURE_ROOT, "features", "baidu"), "fps": 1, "dim": 8576},
}

def parse_timestamp(time_obj):
    if isinstance(time_obj, (int, float)): return float(time_obj)
    if isinstance(time_obj, str):
        if ':' in time_obj:
            parts = time_obj.split(':')
            if len(parts) == 2:
                try: return float(parts[0]) * 60 + float(parts[1])
                except: pass
        try: return float(time_obj)
        except: pass
    return None

def _load_full_features(npy_path):
    if not os.path.exists(npy_path): return None
    try:
        features = np.load(npy_path)
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.squeeze(1)
        return features
    except: return None

def _get_npy_path(path_prefix, half, feature_type):
    config = FEATURE_CONFIG.get(feature_type)
    if not config: return None
    search_path = os.path.join(config["root"], path_prefix, f"{half}*.npy")
    files = glob.glob(search_path)
    return files[0] if files else None

def _extract_feature_window(full_features, timestamp, window_size, fps):
    total_frames = int(window_size * 2 * fps)
    avail = len(full_features)
    center = int(timestamp * fps)
    if center >= avail: center = avail - 1
    if center < 0: center = 0
    start = center - (total_frames // 2)
    end = start + total_frames
    if end > avail:
        end = avail
        start = max(0, end - total_frames)
    elif start < 0:
        start = 0
        end = min(avail, start + total_frames)
    feat = full_features[start:end]
    if feat.shape[0] == 0:
        return np.zeros((1, full_features.shape[1]), dtype=np.float32)
    return feat.astype(np.float32)

class SummaryPairDataset(Dataset):
    def __init__(self, feature_type, data_dir, tokenizer_name="facebook/bart-large", window_size=15, max_text_len=128):
        self.feature_type = feature_type
        self.window_size = window_size
        self.max_text_len = max_text_len
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        
        config = FEATURE_CONFIG.get(feature_type)
        if not config: raise ValueError(f"Unknown feature type: {feature_type}")
        self.fps = config["fps"]
        self.expected_dim = config["dim"]
        
        self.samples = []
        self.load_data(data_dir)

    def load_data(self, data_dir):
        print(f"Loading summary data from {data_dir}")
        json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
        
        TARGET_FILENAMES = {"TARGET_FILENAMES_HERE"}  
        
        cnt = 0
        for file_path in json_files:
            if os.path.basename(file_path) not in TARGET_FILENAMES: continue
            
            parts = re.split(r'[\\/]', file_path)
            if len(parts) < 3: continue
            half = os.path.basename(file_path)[0]
            path_prefix = os.path.join(parts[-3], parts[-2])
            
            with open(file_path, "r", encoding="utf-8") as f:
                try: data = json.load(f).get("annotations", [])
                except: continue
            
            for event in data:
                anchor_time = parse_timestamp(event.get("game_time"))
                if anchor_time is None: continue
                anchor_info = {"path_prefix": path_prefix, "half": half, "time": anchor_time}
                
                for hist in event.get("history", []):
                    hist_time = parse_timestamp(hist.get("history_time"))
                    label_text = hist.get("long-term")
                    if hist_time is None or not label_text: continue
                    hist_info = {"path_prefix": path_prefix, "half": half, "time": hist_time}
                    
                    self.samples.append((anchor_info, hist_info, label_text))
                    cnt += 1
        print(f"Loaded {cnt} summary training pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_info, hist_info, label_text = self.samples[idx]
        
        anchor_vec = self._load_feat(anchor_info)
        hist_vec = self._load_feat(hist_info)
        
        if anchor_vec is None or hist_vec is None: return None
        
        labels = self.tokenizer(
            label_text, 
            max_length=self.max_text_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        label_ids = labels.input_ids.squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        
        time_diff = anchor_info["time"] - hist_info["time"]
        
        return {
            "anchor_feat": torch.tensor(anchor_vec, dtype=torch.float32),
            "hist_feat": torch.tensor(hist_vec, dtype=torch.float32),
            "time_diff": torch.tensor(time_diff, dtype=torch.float32),
            "labels": label_ids
        }

    def _load_feat(self, info):
        try:
            npy_path = _get_npy_path(info["path_prefix"], info["half"], self.feature_type)
            if not npy_path: return None
            full = _load_full_features(npy_path)
            if full is None: return None
            return _extract_feature_window(full, info["time"], self.window_size, self.fps)
        except: return None

def summary_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return None
    
    def pad_video(tensor_list):
        max_len = max([t.shape[0] for t in tensor_list])
        dim = tensor_list[0].shape[1]
        
        padded_feats = []
        masks = []
        
        for t in tensor_list:
            cur_len = t.shape[0]
            pad_len = max_len - cur_len
            
            # Feature Padding
            pad = torch.zeros((pad_len, dim))
            padded_feats.append(torch.cat([t, pad], dim=0))
            
            # Mask Generation (1 for valid, 0 for pad)
            mask = torch.cat([torch.ones(cur_len), torch.zeros(pad_len)], dim=0)
            masks.append(mask)
            
        return torch.stack(padded_feats), torch.stack(masks)

    # Unpack features and masks
    anchors, anchor_mask = pad_video([x["anchor_feat"] for x in batch])
    hists, hist_mask = pad_video([x["hist_feat"] for x in batch])
    
    time_diffs = torch.stack([x["time_diff"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    
    return {
        "anchor_feat": anchors,
        "anchor_mask": anchor_mask, 
        "hist_feat": hists,
        "hist_mask": hist_mask,    
        "time_diff": time_diffs,
        "labels": labels
    }