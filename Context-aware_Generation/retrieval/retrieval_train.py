import torch
import os
import numpy as np
import argparse
import sys
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from retrieval_dataset import RetrievalTripletDataset, retrieval_collate_fn, FEATURE_CONFIG
from retrieval_model import TripletModel

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    
    feature_choices = list(FEATURE_CONFIG.keys()) + ["all"]
    parser.add_argument("--feature_type", type=str, default="all", 
                        choices=feature_choices)
    
    parser.add_argument("--data_dir_root", type=str, default="TEXT_DATA", 
                        )
    parser.add_argument("--save_root", type=str, default="checkpoints_retrieval")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=1e-5) 
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=500)
    

    parser.add_argument("--use_cls", action="store_true", default=False)
    parser.add_argument("--no_cls", action="store_false", dest="use_cls")
    
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def log_to_file(log_str, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(log_str + "\n")
    print(log_str)

def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    all_pos_dists = []
    all_neg_dists = []
    pdist = torch.nn.PairwiseDistance(p=2)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if batch is None: continue 
            
            anchor = batch["anchor_features"].to(device)
            positive = batch["positive_features"].to(device)
            negative = batch["negative_features"].to(device)
            
            anchor_t = batch["anchor_time_diff"].to(device)
            pos_t = batch["positive_time_diff"].to(device)
            neg_t = batch["negative_time_diff"].to(device)
            
            loss, a_vec, p_vec, n_vec = model(
                anchor, positive, negative, 
                anchor_t, pos_t, neg_t
            )
            
            total_val_loss += loss.item()
            
            all_pos_dists.extend(pdist(a_vec, p_vec).cpu().numpy())
            all_neg_dists.extend(pdist(a_vec, n_vec).cpu().numpy())

    avg_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_pos_dist = np.mean(all_pos_dists) if all_pos_dists else 0
    avg_neg_dist = np.mean(all_neg_dists) if all_neg_dists else 0
    
    return avg_loss, avg_pos_dist, avg_neg_dist

def train_single_feature(args, feature_type):

    feat_config = FEATURE_CONFIG[feature_type]
    feature_dim = feat_config["dim"]
    
    suffix = "cls" if args.use_cls else "pool"
    save_dir = os.path.join(args.save_root, f"{feature_type}_{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    log_to_file(f"\n{'='*20} Starting Phase 1 Training: {feature_type} ({suffix}) {'='*20}", log_file)
    log_to_file(f"Feature Dim: {feature_dim}", log_file)
    log_to_file(f"Args: {vars(args)}", log_file)
    
    model = TripletModel(
        feature_dim=feature_dim, 
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        margin=args.margin,
        dropout=args.dropout,
        max_len=args.max_len,
        use_cls=args.use_cls 
    ).to(device)
    
    log_to_file("Loading datasets...", log_file)
    train_dataset = RetrievalTripletDataset(
        data_dir=os.path.join(args.data_dir_root, "train"),
        feature_type=feature_type,
        window_size=args.window_size
    )
    val_dataset = RetrievalTripletDataset(
        data_dir=os.path.join(args.data_dir_root, "val"),
        feature_type=feature_type,
        window_size=args.window_size
    )
    
    if len(train_dataset) == 0:
        log_to_file(f"Error: Dataset for {feature_type} is empty. Skipping.", log_file)
        return

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=retrieval_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        collate_fn=retrieval_collate_fn, num_workers=4, pin_memory=True
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        
        desc = f"[{feature_type}-{suffix}] Epoch {epoch+1}/{args.num_epochs}"
        for batch in tqdm(train_loader, desc=desc):
            if batch is None: continue
                
            optimizer.zero_grad()
            
            anchor = batch["anchor_features"].to(device)
            positive = batch["positive_features"].to(device)
            negative = batch["negative_features"].to(device)
            
            anchor_t = batch["anchor_time_diff"].to(device)
            pos_t = batch["positive_time_diff"].to(device)
            neg_t = batch["negative_time_diff"].to(device)
            
            loss, _, _, _ = model(
                anchor, positive, negative, 
                anchor_t, pos_t, neg_t
            )
            
            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_loss, pos_dist, neg_dist = validate(model, val_loader, device)
        
        log_str = (
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f} | "
            f"Pos Dist={pos_dist:.4f}, Neg Dist={neg_dist:.4f} | "
            f"Delta={neg_dist - pos_dist:.4f}"
        )
        log_to_file(log_str, log_file)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            encoder_path = os.path.join(save_dir, "best_encoder.pth")
            torch.save(model.encoder.state_dict(), encoder_path)
            log_to_file(f">>> New Best Model Saved to {encoder_path}!", log_file)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log_to_file(f"Early stopping triggered after {patience_counter} epochs.", log_file)
                break

    log_to_file(f"Training complete for {feature_type} ({suffix}).\n", log_file)

def main():
    args = parse_args()
    
    if args.feature_type == "all":
        print(f"Training ALL features sequentially: {list(FEATURE_CONFIG.keys())}")
        for feature_type in FEATURE_CONFIG.keys():
            try:
                train_single_feature(args, feature_type)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"!!! Critical Error training {feature_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        print(f"Mode: Training SINGLE feature: {args.feature_type}")
        train_single_feature(args, args.feature_type)

if __name__ == "__main__":
    main()

