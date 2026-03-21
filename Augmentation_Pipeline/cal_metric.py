import torch
import os
import json
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import random

random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ann_root_path = "TEXT_PATH"

model_types = [
    "Visual Feature types here",
]

def init_container():
    return {
        "refs": [],
        "hyps": [],
        "METEOR": [],
        "rougeL": []
    }

results = {
    model: {
        "description": init_container(),
        "short_term": init_container(),
    }
    for model in model_types
}

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

json_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(ann_root_path)
    for file in files if file.endswith(".json")
]

print(f"Found {len(json_files)} JSON files.")

for file_path in tqdm(json_files, desc="Processing JSON files"):

    file_name = os.path.basename(file_path)

    matched_model = None
    for model in model_types:
        if model in file_name:
            matched_model = model
            break

    if matched_model is None:
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if "ori_text" not in entry:
            continue

        gt = entry["ori_text"].strip()
        ref_tokens = [gt.split()]

        if "description" in entry and entry["description"].strip() != "":
            desc = entry["description"].strip()
            desc_tokens = desc.split()

            results[matched_model]["description"]["refs"].append(ref_tokens)
            results[matched_model]["description"]["hyps"].append(desc_tokens)

            results[matched_model]["description"]["METEOR"].append(
                meteor_score(ref_tokens, desc_tokens)
            )

            rouge_res = rouge.score(gt, desc)
            results[matched_model]["description"]["rougeL"].append(
                rouge_res["rougeL"].fmeasure
            )

        if "short-term" in entry and entry["short-term"].strip() != "":
            st = entry["short-term"].strip()
            st_tokens = st.split()

            results[matched_model]["short_term"]["refs"].append(ref_tokens)
            results[matched_model]["short_term"]["hyps"].append(st_tokens)

            results[matched_model]["short_term"]["METEOR"].append(
                meteor_score(ref_tokens, st_tokens)
            )

            rouge_res = rouge.score(gt, st)
            results[matched_model]["short_term"]["rougeL"].append(
                rouge_res["rougeL"].fmeasure
            )


smooth = SmoothingFunction().method1
final_scores = {}

def compute_scores(container):
    refs = container["refs"]
    hyps = container["hyps"]

    if len(refs) == 0:
        return None

    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "BLEU-1": round(bleu1 * 100, 2),
        "BLEU-4": round(bleu4 * 100, 2),
        "METEOR": round(np.mean(container["METEOR"]) * 100, 2),
        "rougeL": round(np.mean(container["rougeL"]) * 100, 2),
    }


for model in model_types:
    desc_scores = compute_scores(results[model]["description"])
    st_scores = compute_scores(results[model]["short_term"])

    final_scores[model] = {
        "description_scores": desc_scores,
        "short_term_scores": st_scores,
    }

with open("evaluation.json", "w", encoding="utf-8") as f:
    json.dump(final_scores, f, indent=4, ensure_ascii=False)

print("\n Results saved to evaluation.json\n")
