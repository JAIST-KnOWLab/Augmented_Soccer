import torch
import os
import argparse
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from rouge import Rouge 
from nltk.translate.bleu_score import corpus_bleu

from summary_dataset import SummaryPairDataset, summary_collate_fn, FEATURE_CONFIG

from summary_model import VisualBartQFormerSummarizer

def parse_args():
    parser = argparse.ArgumentParser()
    
    feature_choices = list(FEATURE_CONFIG.keys()) + ["all"]
    parser.add_argument("--feature_type", type=str, default="all", choices=feature_choices)
    
    parser.add_argument("--data_root", type=str, default="TEXT_PATH")
    parser.add_argument("--save_dir", type=str, default="MODEL_SAVE_PATH")
    
    parser.add_argument("--batch_size", type=int, default=18) 
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5) 
    parser.add_argument("--dropout", type=float, default=0.1)
    
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--qformer_layers", type=int, default=2)
    
    args = parser.parse_args()
    return args

def train_single_feature(args, feature_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_save_dir = os.path.join(args.save_dir, feature_type)
    os.makedirs(feature_save_dir, exist_ok=True)
    
    log_file = os.path.join(feature_save_dir, "training_log.txt")
    best_record_file = os.path.join(feature_save_dir, "best_results.txt")
    
    print(f"\n{'='*20} Loading Data for {feature_type} {'='*20}")
    train_ds = SummaryPairDataset(feature_type, os.path.join(args.data_root, "train"))
    val_ds = SummaryPairDataset(feature_type, os.path.join(args.data_root, "val"))
    
    if len(train_ds) == 0:
        print(f"Error: No training data found for {feature_type}. Skipping.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=summary_collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=summary_collate_fn, num_workers=4)
    
    feat_dim = FEATURE_CONFIG[feature_type]["dim"]
    print(f"Initializing Q-Former Generator (Input Dim: {feat_dim})...")
    
    model = VisualBartQFormerSummarizer(
        feature_dim=feat_dim,          
        num_query_tokens=args.num_query_tokens,
        bart_model_name="facebook/bart-base",
        qformer_hidden_layers=args.qformer_layers,
        dropout=args.dropout
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    rouge = Rouge()
    best_rouge_l = 0.0
    
    with open(log_file, "w") as f:
        f.write("Epoch\tTrain Loss\tVal Loss\tROUGE-L\tBLEU-1\n")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        desc = f"[{feature_type}] Epoch {epoch+1}"
        pbar = tqdm(train_loader, desc=desc)
        
        for batch in pbar:
            if batch is None: continue
            optimizer.zero_grad()
            
            anchor = batch["anchor_feat"].to(device)
            hist = batch["hist_feat"].to(device)
            td = batch["time_diff"].to(device)
            lbl = batch["labels"].to(device)
            
            anchor_mask = batch["anchor_mask"].to(device)
            hist_mask = batch["hist_mask"].to(device)
            
            loss, _ = model(
                anchor, hist, td, labels=lbl,
                anchor_mask=anchor_mask, 
                hist_mask=hist_mask      
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        references = []
        
        print(f"Running Validation for {feature_type}...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if batch is None: continue
                anchor = batch["anchor_feat"].to(device)
                hist = batch["hist_feat"].to(device)
                td = batch["time_diff"].to(device)
                lbl = batch["labels"].to(device)
                
                anchor_mask = batch["anchor_mask"].to(device)
                hist_mask = batch["hist_mask"].to(device)
                
                loss, _ = model(
                    anchor, hist, td, labels=lbl,
                    anchor_mask=anchor_mask,
                    hist_mask=hist_mask
                )
                val_loss += loss.item()
                
                #Generation
                generated_ids = model.generate(
                    anchor, hist, td, 
                    max_length=100,
                    anchor_mask=anchor_mask, 
                    hist_mask=hist_mask      
                )
                
                decoded_preds = train_ds.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                lbl[lbl == -100] = train_ds.tokenizer.pad_token_id
                decoded_labels = train_ds.tokenizer.batch_decode(lbl, skip_special_tokens=True)
                
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
        
        avg_val_loss = val_loss / len(val_loader)
        
        try:
            scores = rouge.get_scores(predictions, references, avg=True)
            rouge_l = scores['rouge-l']['f']
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            rouge_l = 0.0

        try:
            refs_tokens = [[r.strip().split()] for r in references]
            preds_tokens = [p.strip().split() for p in predictions]
            bleu_1 = corpus_bleu(refs_tokens, preds_tokens, weights=(1.0, 0, 0, 0))
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            bleu_1 = 0.0
            
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, ROUGE-L {rouge_l:.4f}, BLEU-1 {bleu_1:.4f}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{rouge_l:.4f}\t{bleu_1:.4f}\n")
        
        if rouge_l > best_rouge_l:
            best_rouge_l = rouge_l
            save_path = os.path.join(feature_save_dir, "best_generator.pth")
            torch.save(model.state_dict(), save_path)
            
            with open(best_record_file, "w") as f:
                f.write(f"Best ROUGE-L: {best_rouge_l:.4f}\n")
                f.write(f"Corresponding BLEU-1: {bleu_1:.4f}\n")
                f.write(f"Achieved at Epoch: {epoch+1}\n")
                f.write(f"Val Loss: {avg_val_loss:.4f}\n")
                
            print(f"Saved Best Model to {save_path}! (New Best ROUGE-L: {best_rouge_l:.4f})")
    
    del model
    del optimizer
    torch.cuda.empty_cache()
    print(f"Training for {feature_type} completed.\n")

def main():
    args = parse_args()
    
    if args.feature_type == "all":
        print(f"Mode: Training ALL features sequentially: {list(FEATURE_CONFIG.keys())}")
        for feature_type in FEATURE_CONFIG.keys():
            try:
                train_single_feature(args, feature_type)
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