import os
import json
import glob
import random
import numpy as np
import torch
import re
from torch.utils.data import Dataset
import warnings


FEATURE_ROOT = "FEATURE_ROOT_PATH_HERE" 

FEATURE_CONFIG = {
    "baidu": {
        "root": os.path.join(FEATURE_ROOT, "features", "baidu"),
        "fps": 1,
        "dim": 8576
    },
    "c3d_2fps": {
        "root": os.path.join(FEATURE_ROOT, "features", "c3d_2fps"),
        "fps": 2,
        "dim": 4096
    },
    "i3d_2fps": {
        "root": os.path.join(FEATURE_ROOT, "features", "i3d_2fps"),
        "fps": 2,
        "dim": 1024
    },
    "resnet_2fps_512": {
        "root": os.path.join(FEATURE_ROOT, "features", "resnet_2fps_512"),
        "fps": 2,
        "dim": 512
    },
    "resnet_5fps": {
        "root": os.path.join(FEATURE_ROOT, "features", "resnet_5fps"),
        "fps": 5,
        "dim": 2048
    },
    "clip": {
        "root": os.path.join(FEATURE_ROOT, "features", "clip"),
        "fps": 2,
        "dim": 512 
    },
}

_feature_cache = {}

def parse_timestamp(time_obj):

    if isinstance(time_obj, (int, float)):
        return float(time_obj)
    
    if isinstance(time_obj, str):
        if ':' in time_obj:
            parts = time_obj.split(':')
            if len(parts) == 2:
                try:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                except ValueError:
                    pass
        try:
            return float(time_obj)
        except ValueError:
            pass
            
    return None

def _load_full_features(npy_path):
    if npy_path in _feature_cache:
        return _feature_cache[npy_path]
    
    if not os.path.exists(npy_path):
        warnings.warn(f"Not found: {npy_path}", RuntimeWarning)
        return None
        
    try:
        features = np.load(npy_path)
        
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.squeeze(1)
            
        _feature_cache[npy_path] = features
        return features
    except Exception as e:
        warnings.warn(f"Load failed: {npy_path}, Error: {e}", RuntimeWarning)
        return None

def _get_npy_path(path_prefix, half, feature_type):
    config = FEATURE_CONFIG.get(feature_type)
    
    if not config:
        warnings.warn(f"UNKNOW FEATURE_TYPE: {feature_type}", RuntimeWarning)
        return None
    
    root_folder = config["root"]
    search_path = os.path.join(root_folder, path_prefix, f"{half}*.npy")
    
    files_found = glob.glob(search_path)
    if not files_found:
        return None
    
    return files_found[0]

def _extract_feature_window(full_features, timestamp, window_size, fps):
    total_frames_in_window = int(window_size * 2 * fps)
    total_frames_available = len(full_features)
    
    center_frame = int(timestamp * fps)
    
    if center_frame >= total_frames_available:
        center_frame = total_frames_available - 1
    if center_frame < 0: 
        center_frame = 0

    start_frame = center_frame - (total_frames_in_window // 2)
    end_frame = start_frame + total_frames_in_window

    if end_frame > total_frames_available:
        end_frame = total_frames_available
        start_frame = max(0, end_frame - total_frames_in_window)
    elif start_frame < 0:
        start_frame = 0
        end_frame = min(total_frames_available, start_frame + total_frames_in_window)
        
    feature_window = full_features[start_frame:end_frame]
    
    if feature_window.shape[0] == 0:
        feature_dim = full_features.shape[1]
        return np.zeros((1, feature_dim), dtype=np.float32)
        
    return feature_window.astype(np.float32)

class RetrievalTripletDataset(Dataset):
    def __init__(
        self,
        feature_type,
        data_dir="TEXT_DATA_DIR_HERE", 
        window_size=15 #half
    ):
        super().__init__()
        self.feature_type = feature_type
        self.window_size = window_size
        self.data_dir = data_dir
        config = FEATURE_CONFIG.get(feature_type)
        if config is None:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        self.fps = config["fps"]
        self.expected_dim = config["dim"]
        
        self.triplets = []
        self.load_data(data_dir)

    def load_data(self, data_dir):
        print(f"Loading data from {data_dir}, feature type: {self.feature_type}...")
        
        json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
        TARGET_FILENAMES = {"JSON files here"}
        
        event_info_by_id = {} 
        positive_map = {}     
        anchor_ids = []       
        
   
        match_anchors_map = {}

        event_counter = 0

        for file_path in json_files:
            filename = os.path.basename(file_path)
            if filename not in TARGET_FILENAMES: continue

            parts = re.split(r'[\\/]', file_path)
            if len(parts) < 3: continue
            half = filename[0] 
            path_prefix = os.path.join(parts[-3], parts[-2]) 
            
            match_key = (path_prefix, half)
            if match_key not in match_anchors_map:
                match_anchors_map[match_key] = []
            
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f).get("annotations", [])
                except json.JSONDecodeError:
                    continue
            
            # 1.  Anchor
            for event in data:
                raw_time = event.get("game_time")
                anchor_time = parse_timestamp(raw_time)
                
                if anchor_time is None: continue
                
                anchor_id = f"evt_{event_counter}"
                event_counter += 1
                
                anchor_info = {
                    "path_prefix": path_prefix, 
                    "half": half, 
                    "time": anchor_time
                }
                event_info_by_id[anchor_id] = anchor_info
                anchor_ids.append(anchor_id)
                
           
                match_anchors_map[match_key].append(anchor_id)
                
                if anchor_id not in positive_map:
                    positive_map[anchor_id] = []

                # 2. process Anchor History (Positives)
                for history_event in event.get("history", []):
                    raw_hist_time = history_event.get("history_time")
                    history_time = parse_timestamp(raw_hist_time)
                    
                    if history_time is None: continue
                    
                    pos_id = f"hist_{event_counter}"
                    event_counter += 1
                    
                    pos_info = {
                        "path_prefix": path_prefix,
                        "half": half,
                        "time": history_time
                    }
                    event_info_by_id[pos_id] = pos_info
                    
                    positive_map[anchor_id].append(pos_id)
        
        if not anchor_ids:
            print("000")
            return

        # negative sampling
        global_negative_pool = anchor_ids 

        for anchor_id in anchor_ids:
            positive_ids = positive_map[anchor_id]
            if not positive_ids: continue
                
            anchor_info = event_info_by_id[anchor_id]
            
            current_match_key = (anchor_info["path_prefix"], anchor_info["half"])
            match_pool = match_anchors_map.get(current_match_key, [])
            
            use_global_pool = False
            if len(match_pool) <= 1:
                use_global_pool = True
                target_pool = global_negative_pool
            else:
                target_pool = match_pool

            for pos_id in positive_ids:
                pos_info = event_info_by_id[pos_id]
                
                neg_info = None
                for _ in range(10): 
                    neg_id = random.choice(target_pool)
                    
                    if neg_id == anchor_id: continue
                    
                    candidate_neg_info = event_info_by_id[neg_id]
                    
                    # 检查时间冲突 (容差 10s)，避免选到同一时刻的事件作为负例
                    # 如果是同比赛采样，path 和 half 肯定相同，主要检查 time
                    if (candidate_neg_info["path_prefix"] == anchor_info["path_prefix"] and
                        candidate_neg_info["half"] == anchor_info["half"] and
                        abs(candidate_neg_info["time"] - pos_info["time"]) < 10.0): 
                        continue
                    
                    neg_info = candidate_neg_info
                    break
                
                # 如果在同比赛内没找到合适的负例，尝试全局回退
                if neg_info is None and not use_global_pool:
                     for _ in range(10):
                        neg_id = random.choice(global_negative_pool)
                        if neg_id == anchor_id: continue
                        candidate_neg_info = event_info_by_id[neg_id]
                        # 全局回退时也要防止万一随机到了同一场比赛的同一时刻
                        if (candidate_neg_info["path_prefix"] == anchor_info["path_prefix"] and
                            candidate_neg_info["half"] == anchor_info["half"] and
                            abs(candidate_neg_info["time"] - pos_info["time"]) < 10.0): 
                            continue
                        neg_info = candidate_neg_info
                        break

                if neg_info:
                    self.triplets.append( (anchor_info, pos_info, neg_info) )
                
        print(f"Loaded {len(self.triplets)} triplets for retrieval training.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_info, positive_info, negative_info = self.triplets[idx]
        
        anchor_seq = self._load_and_extract(anchor_info)
        positive_seq = self._load_and_extract(positive_info)
        negative_seq = self._load_and_extract(negative_info)

        if anchor_seq is None or positive_seq is None or negative_seq is None:
            return None

        pos_time_diff = anchor_info['time'] - positive_info['time']
        neg_time_diff = anchor_info['time'] - negative_info['time']
        anchor_time_diff = 0.0 

        return {
            "anchor_features": torch.tensor(anchor_seq, dtype=torch.float32),
            "positive_features": torch.tensor(positive_seq, dtype=torch.float32),
            "negative_features": torch.tensor(negative_seq, dtype=torch.float32),
            
            "anchor_time_diff": torch.tensor(anchor_time_diff, dtype=torch.float32),
            "positive_time_diff": torch.tensor(pos_time_diff, dtype=torch.float32),
            "negative_time_diff": torch.tensor(neg_time_diff, dtype=torch.float32)
        }

    def _load_and_extract(self, info):
        path_prefix = info["path_prefix"]
        half = info["half"]
        timestamp = info["time"]
        
        npy_path = _get_npy_path(path_prefix, half, self.feature_type)
        if npy_path is None: return None
            
        full_features = _load_full_features(npy_path)
        if full_features is None: return None
        
        if full_features.shape[1] != self.expected_dim:
            pass 

        feature_window = _extract_feature_window(
            full_features,
            timestamp,
            self.window_size,
            self.fps
        )
        return feature_window
     
def retrieval_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None

    keys = batch[0].keys()
    padded_batch = {key: [] for key in keys}

    video_keys = ["anchor_features", "positive_features", "negative_features"]
    
    max_lens = {}
    for key in video_keys:
        max_lens[key] = max(item[key].shape[0] for item in batch)

    any_video_key = video_keys[0]
    feature_dim = batch[0][any_video_key].shape[1]
    
    for item in batch:
        for key in keys:
            data = item[key]
            
            if key in video_keys:
                seq_len = data.shape[0]
                max_len = max_lens[key]
                
                if seq_len < max_len:
                    padding = torch.zeros((max_len - seq_len, feature_dim), dtype=torch.float32)
                    padded_seq = torch.cat([data, padding], dim=0)
                else:
                    padded_seq = data
                padded_batch[key].append(padded_seq)
            else:
                padded_batch[key].append(data)

    try:
        final_batch = {key: torch.stack(val) for key, val in padded_batch.items()}
    except Exception as e:
        print(f"Stacking Error: {e}")
        return None

    return final_batch