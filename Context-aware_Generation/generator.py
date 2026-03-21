import json
import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm
from transformers import BartTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "retrieval"))
sys.path.append(os.path.join(current_dir, "generator"))


from retrieval_model import SequenceEncoder as RetrievalModel
from summary_model import VisualBartQFormerSummarizer      


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

PRED_KEY_MAPPING = {
    "baidu": "baidu",
    "c3d": "c3d_2fps",
    "clip": "clip",
    "resnet2": "resnet_2fps_512",
    "resnet5": "resnet_5fps",
    "i3d": "i3d_2fps"
}

RETRIEVAL_CKPT_ROOT = os.path.join("retrieval", "checkpoints_retrieval")
GENERATOR_CKPT_ROOT = os.path.join("generator", "checkpoints_summary_qformer")



class WrapperModel:
    def __init__(self):
        self.retrievers = {}
        self.generators = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading BartTokenizer...")
        try:
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        except Exception:
            print("Failed to load 'facebook/bart-large', trying 'facebook/bart-base'")
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def get_models(self, feature_type):
        if feature_type not in FEATURE_CONFIG:
            return None, None

        # Load Retrieval
        if feature_type not in self.retrievers:
            ckpt_path = os.path.join(RETRIEVAL_CKPT_ROOT, feature_type, "best_encoder.pth")
            if os.path.exists(ckpt_path):
                input_dim = FEATURE_CONFIG[feature_type]['dim']
                
                model = RetrievalModel(
                    feature_dim=input_dim,
                    embed_dim=768,      
                    n_heads=8,          
                    n_layers=4,         
                    dropout=0.1,
                    max_len=5000,
                    use_cls=False       
                )
                
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                
                try:
                    model.load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    model.load_state_dict(state_dict, strict=False)
                
                model.to(self.device)
                model.eval()
                self.retrievers[feature_type] = model
            else:
                return None, None

        # Load Generator
        if feature_type not in self.generators:
            ckpt_path = os.path.join(GENERATOR_CKPT_ROOT, feature_type, "best_generator.pth")
            if os.path.exists(ckpt_path):
                input_dim = FEATURE_CONFIG[feature_type]['dim']
                
                model = VisualBartQFormerSummarizer(
                    feature_dim=input_dim,
                    num_query_tokens=32,
                    bart_model_name="facebook/bart-base", 
                    qformer_hidden_layers=2
                ) 
                
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device)
                model.eval()
                self.generators[feature_type] = model
            else:
                return None, None

        return self.retrievers[feature_type], self.generators[feature_type]

    def run_retrieval(self, model, query_feat, candidate_feats, time_diffs_ms):
        if not isinstance(query_feat, torch.Tensor):
            query_feat = torch.tensor(query_feat, device=self.device, dtype=torch.float32)
        if len(query_feat.shape) == 1:
            query_feat = query_feat.unsqueeze(0).unsqueeze(0)
        elif len(query_feat.shape) == 2:
            query_feat = query_feat.unsqueeze(0)

        query_time = torch.tensor([0.0], device=self.device, dtype=torch.float32)

        if not isinstance(candidate_feats, torch.Tensor):
            candidate_feats = torch.tensor(candidate_feats, device=self.device, dtype=torch.float32)
        if not isinstance(time_diffs_ms, torch.Tensor):
            cand_times = torch.tensor(time_diffs_ms, device=self.device, dtype=torch.float32)
        else:
            cand_times = time_diffs_ms.to(self.device)

        with torch.no_grad():
            query_emb = model(query_feat, query_time) 
            
            cand_embs = []
            batch_size = 32
            for i in range(0, len(candidate_feats), batch_size):
                batch_feat = candidate_feats[i : i + batch_size]
                batch_time = cand_times[i : i + batch_size]
                if len(batch_feat.shape) == 2: 
                     batch_feat = batch_feat.unsqueeze(0)
                out = model(batch_feat, batch_time)
                cand_embs.append(out)
            
            if cand_embs:
                cand_embs = torch.cat(cand_embs, dim=0)
            else:
                return candidate_feats[0].cpu().numpy(), 0

            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            cand_embs = torch.nn.functional.normalize(cand_embs, p=2, dim=1)
            
            scores = torch.matmul(cand_embs, query_emb.t()).squeeze()
            if scores.ndim == 0:
                best_idx = 0
            else:
                best_idx = torch.argmax(scores).item()
            
        return candidate_feats[best_idx].cpu().numpy(), best_idx

    def run_generation(self, model, anchor_feat, hist_feat, time_diff_ms):
        if not isinstance(anchor_feat, torch.Tensor):
            anchor_input = torch.tensor(anchor_feat, device=self.device, dtype=torch.float32)
        else:
            anchor_input = anchor_feat
        if anchor_input.ndim == 2: anchor_input = anchor_input.unsqueeze(0)

        if not isinstance(hist_feat, torch.Tensor):
            hist_input = torch.tensor(hist_feat, device=self.device, dtype=torch.float32)
        else:
            hist_input = hist_feat
        if hist_input.ndim == 2: hist_input = hist_input.unsqueeze(0)
            
        anchor_mask = (anchor_input.abs().sum(dim=-1) > 1e-6).long().to(self.device)
        hist_mask = (hist_input.abs().sum(dim=-1) > 1e-6).long().to(self.device)
        t_input = torch.tensor([time_diff_ms], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            try:
                generated_ids = model.generate(
                    anchor_raw=anchor_input, 
                    hist_raw=hist_input, 
                    time_diff=t_input,
                    max_length=100,
                    anchor_mask=anchor_mask, 
                    hist_mask=hist_mask      
                )
                decoded_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return decoded_text.strip()
            except Exception as e:
                print(f"Gen Error: {e}")
                return "Generation failed."


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
    return 0.0

def load_video_features(feature_type, relative_dir, half):
    config = FEATURE_CONFIG.get(feature_type)
    if not config: return None

    search_dir = os.path.join(config["root"], relative_dir)
    pattern = os.path.join(search_dir, f"{half}_*.npy")
    
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    feat_path = files[0]
    
    try:
        data = np.load(feat_path)
        if len(data.shape) == 3: data = data.squeeze(1) 
        return data
    except Exception as e:
        print(f"Error loading {feat_path}: {e}")
        return None

def extract_feature_window(full_features, timestamp_sec, fps, window_size=15):
    if full_features is None: return None
    
    total_frames_in_window = int(window_size * 2 * fps)
    total_frames_available = len(full_features)
    center_frame = int(timestamp_sec * fps)
    
    if center_frame >= total_frames_available: center_frame = total_frames_available - 1
    if center_frame < 0: center_frame = 0

    start_frame = center_frame - (total_frames_in_window // 2)
    end_frame = start_frame + total_frames_in_window

    if end_frame > total_frames_available:
        end_frame = total_frames_available
        start_frame = max(0, end_frame - total_frames_in_window)
    elif start_frame < 0:
        start_frame = 0
        end_frame = min(total_frames_available, start_frame + total_frames_in_window)
        
    feature_window = full_features[start_frame:end_frame]
    
    if feature_window.shape[0] < total_frames_in_window:
        pad_len = total_frames_in_window - feature_window.shape[0]
        padding = np.zeros((pad_len, full_features.shape[1]), dtype=feature_window.dtype)
        feature_window = np.concatenate([feature_window, padding], axis=0)
        
    return feature_window


def main():
    base_dir = os.path.join("TEXT_DATA", "test")
    file_pattern = os.path.join(base_dir, "**", "*_long-term_*.json")
    json_files = glob.glob(file_pattern, recursive=True)

    if not json_files:
        print(f"No JSON files found in {base_dir}")
        return

    print(f"Found {len(json_files)} files.")
    model_manager = WrapperModel()

    for json_file in json_files:
        print(f"\nProcessing: {os.path.basename(json_file)}")
        

        dir_path = os.path.dirname(json_file)
        
 
        relative_dir = os.path.relpath(dir_path, base_dir)
        
        filename = os.path.basename(json_file)
        
        if filename.startswith("1"):
            half = "1"
            game_filename = "1_game.json"
        elif filename.startswith("2"):
            half = "2"
            game_filename = "2_game.json"
        else:
            print(f"  Skipping {filename}: unknown half.")
            continue
            
        game_json_path = os.path.join(dir_path, game_filename)
        if not os.path.exists(game_json_path):
            print(f"  Game file not found: {game_json_path}")
            continue
            
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(game_json_path, 'r', encoding='utf-8') as gf:
            game_data = json.load(gf)

        annotations = data.get("annotations", [])      # Anchor Events
        game_annotations = game_data.get("annotations", []) # Candidate Events
        
        anchor_events_map = {} 
        candidate_events_map = {} 
        
        needed_config_types = set()
        for event in annotations:
            for k in event.get("prediction", {}).keys():
                if k.endswith("(SCORE)"):
                    raw_key = k.replace("(SCORE)", "")
                    config_key = PRED_KEY_MAPPING.get(raw_key)
                    if config_key and config_key in FEATURE_CONFIG:
                        needed_config_types.add(config_key)
        
        for f_type in needed_config_types:
            full = load_video_features(f_type, relative_dir, half)
            
            if full is None: 
                # print(f"  [Warn] Failed to load features for {f_type}")
                continue 
            
            fps = FEATURE_CONFIG[f_type]["fps"]
            
            # (A) Anchors
            anchor_list = []
            for idx, event in enumerate(annotations):
                t_str = event.get("game_time", "00:00")
                t_sec = parse_timestamp(t_str)
                feat = extract_feature_window(full, t_sec, fps, window_size=15)
                anchor_list.append({
                    "idx": idx, 
                    "feat": feat, 
                    "time_sec": t_sec
                })
            anchor_events_map[f_type] = anchor_list
            
            # (B) Candidates
            cand_list = []
            for idx, event in enumerate(game_annotations):
                t_str = event.get("game_time", "00:00")
                t_sec = parse_timestamp(t_str)
                feat = extract_feature_window(full, t_sec, fps, window_size=15)
                cand_list.append({
                    "idx": idx, 
                    "feat": feat, 
                    "time_sec": t_sec
                })
            candidate_events_map[f_type] = cand_list

        changed_count = 0
        for i, event in tqdm(enumerate(annotations), total=len(annotations), desc="Generating"):
            predictions = event.get("prediction", {})
            target_keys = [k for k in predictions.keys() if k.endswith("(SCORE)")]
            
            for key in target_keys:
                raw_key = key.replace("(SCORE)", "")
                feature_type = PRED_KEY_MAPPING.get(raw_key)
                
                if not feature_type or \
                   feature_type not in anchor_events_map or \
                   feature_type not in candidate_events_map: 
                    continue
                
                retriever, generator = model_manager.get_models(feature_type)
                if not retriever or not generator: continue
                
                current_info = anchor_events_map[feature_type][i]
                current_feat = current_info["feat"]
                current_time = current_info["time_sec"]
                
                candidates_feat = []
                candidates_time_diffs = []
                
                for item in candidate_events_map[feature_type]:
                    if item["time_sec"] >= current_time: 
                        continue
                    
                    candidates_feat.append(item["feat"])
                    diff_sec = current_time - item["time_sec"]
                    candidates_time_diffs.append(diff_sec * 1000.0)
                
                if not candidates_feat: continue
                
                candidates_np = np.array(candidates_feat)
                candidates_diffs_np = np.array(candidates_time_diffs)
                
                # Retrieval
                best_feat, best_idx = model_manager.run_retrieval(
                    retriever, current_feat, candidates_np, candidates_diffs_np
                )
                
                # Generator
                best_diff_ms = candidates_diffs_np[best_idx]
                long_text = model_manager.run_generation(
                    generator, current_feat, best_feat, best_diff_ms
                )
                
                # Update
                val = predictions[key]
                entry = {"long-term": long_text}
                
                if isinstance(val, list):
                    if len(val) > 0 and isinstance(val[-1], dict) and "long-term" in val[-1]:
                        val[-1] = entry
                    else:
                        val.append(entry)
                else:
                    predictions[key] = [val, entry]
                
                changed_count += 1

        if changed_count > 0:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"  -> Saved {changed_count} captions.")
        else:
            print("  -> No updates.")

if __name__ == "__main__":
    main()