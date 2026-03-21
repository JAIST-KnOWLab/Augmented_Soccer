import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from dataset import Short_Term_Dataset  
from MMTBART_model import Frame_Predict_Event_Model  
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
import os



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    special_tokens = ["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])", "[STADIUM]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer



def eval_cider(predicted_captions, gt_captions, tokenizer):
    cider_evaluator = Cider()
    predicted_captions_dict = {
        str(i): [pred.strip() if isinstance(pred, str) else tokenizer.decode(pred, skip_special_tokens=True).strip()]
        for i, pred in enumerate(predicted_captions)
    }
    gt_captions_dict = {
        str(i): [gt.strip() if isinstance(gt, str) else tokenizer.decode(gt, skip_special_tokens=True).strip()]
        for i, gt in enumerate(gt_captions)
    }
    avg_cider_score, _ = cider_evaluator.compute_score(predicted_captions_dict, gt_captions_dict)
    return avg_cider_score


def eval_bleu(predicted_captions, gt_captions, tokenizer):
    bleu_evaluator = Bleu(4)
    predicted_captions_dict = {
        str(i): [pred.strip() if isinstance(pred, str) else tokenizer.decode(pred, skip_special_tokens=True).strip()]
        for i, pred in enumerate(predicted_captions)
    }
    gt_captions_dict = {
        str(i): [gt.strip() if isinstance(gt, str) else tokenizer.decode(gt, skip_special_tokens=True).strip()]
        for i, gt in enumerate(gt_captions)
    }
    scores, _ = bleu_evaluator.compute_score(gt_captions_dict, predicted_captions_dict)
    bleu4 = scores[3]  # BLEU-4
    return bleu4



def train_one_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    train_pbar = tqdm(train_loader, desc="Training")
    for batch in train_pbar:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    return total_loss / len(train_loader)


def validate(model, val_loader, tokenizer):
    model.eval()
    predicted_texts, gt_texts = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            predicted, gt = model(batch, validating=True)
            predicted_texts.extend(predicted)
            gt_texts.extend(gt)

    cider_score = eval_cider(predicted_texts, gt_texts, tokenizer)
    bleu4_score = eval_bleu(predicted_texts, gt_texts, tokenizer)

    print(f"\nValidation Results — CIDEr: {cider_score:.4f}, BLEU-4: {bleu4_score:.4f}\n")

    
    for i in range(min(3, len(predicted_texts))):
        pred_text = predicted_texts[i] if isinstance(predicted_texts[i], str) else tokenizer.decode(predicted_texts[i], skip_special_tokens=True).strip()
        gt_text = gt_texts[i] if isinstance(gt_texts[i], str) else tokenizer.decode(gt_texts[i], skip_special_tokens=True).strip()
        print(f"[Pred] {pred_text}")
        print(f"[GT]   {gt_text}")
        print("")

    return cider_score, bleu4_score


def save_model(model, file_path):
    torch.save(model.cpu().state_dict(), file_path)
    model.to(model.device)



def train(args):
    set_seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer_name)

    train_dataset = Short_Term_Dataset(
        feature_root_path=args.feature_root,
        ann_root_path=os.path.join(args.data_path, "train"),
        window=args.window,
        fps=args.fps,
        tokenizer_name=args.tokenizer_name
    )

    val_dataset = Short_Term_Dataset(
        feature_root_path=args.feature_root,
        ann_root_path=os.path.join(args.data_path, "val"),
        window=args.window,
        fps=args.fps,
        tokenizer_name=args.tokenizer_name
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              shuffle=True, collate_fn=train_dataset.collater, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers,
                            shuffle=False, collate_fn=val_dataset.collater, pin_memory=True)

    model = Frame_Predict_Event_Model(
        lm_ckpt=args.model_checkpoint,
        tokenizer_ckpt=args.tokenizer_name,
        window=args.window,
        fps=args.fps,
        feature_dim=args.feature_dim,
        device=args.device
    ).to(args.device)

    if args.continue_train:
        model.load_state_dict(torch.load(args.load_ckpt))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_cider = -1.0  
    best_model_path = os.path.join(args.model_output_dir, "best_model.pth")

    for epoch in range(args.num_epoch):

        train_loss = train_one_epoch(model, train_loader, optimizer)

        val_cider, val_bleu = validate(model, val_loader, tokenizer)

        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} | CIDEr: {val_cider:.4f} | BLEU-4: {val_bleu:.4f}")

        #  10 epoch
        if (epoch + 1) % 10 == 0:
            snapshot_path = os.path.join(args.model_output_dir, f"model_epoch_{epoch+1}.pth")
            save_model(model, snapshot_path)
            print(f"Saved checkpoint: {snapshot_path}")

        if val_cider > best_cider:
            best_cider = val_cider
            save_model(model, best_model_path)
            print(f"New best model saved (CIDEr={best_cider:.4f}) → {best_model_path}")




if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="TEXT_PATH")
    parser.add_argument("--feature_root", type=str, default="FEATURE_ROOT_PATH")
    parser.add_argument("--model_checkpoint", type=str, default="facebook/bart-large")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-large")

    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--val_batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.02)

    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--feature_dim", type=int, default=2048)

    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--load_ckpt", type=str, default="CKPT_PATH")
    parser.add_argument("--model_output_dir", type=str, default="OUTPUT_DIR")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.model_output_dir, exist_ok=True)
    train(args)
