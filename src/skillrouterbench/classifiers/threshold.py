"""Threshold optimization for guardrail (recall >= 0.95) and router abstain rule."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_jsonl(path: Path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Optimize classifier thresholds")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--data_dir", type=str, default="data/cmg")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Guardrail Threshold Optimization
    guardrail_path = Path(args.model_dir) / "guardrail"
    g_tokenizer = AutoTokenizer.from_pretrained(str(guardrail_path))
    g_model = AutoModelForSequenceClassification.from_pretrained(str(guardrail_path))
    g_model.eval()

    # Load temperature
    g_calib_file = out_dir / "calibration_guardrail.json"
    g_temp = 1.0
    if g_calib_file.exists():
        with open(g_calib_file, "r") as f:
            g_temp = json.load(f)["temperature"]

    val_data = load_jsonl(Path(args.data_dir) / "val.jsonl")
    texts = [x["text"] for x in val_data]
    labels = np.array([1 if x["guardrail"] == "needs_human_review" else 0 for x in val_data])

    all_probs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        inputs = g_tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = g_model(**inputs)
            logits = outputs.logits / g_temp
            probs = torch.softmax(logits, dim=-1).numpy()
            all_probs.append(probs)

    probs_arr = np.vstack(all_probs)
    review_probs = probs_arr[:, 1]  # probability of needs_human_review

    thresholds = np.linspace(0.1, 0.95, 86)
    best_threshold = 0.5
    min_review_rate = 1.0
    sweep_results = []

    for t in thresholds:
        # If predicted probability of needs_human_review >= t, route to review
        preds = (review_probs >= t).astype(int)
        # True positives (actual review correctly caught)
        tp = np.sum((preds == 1) & (labels == 1))
        actual_positives = np.sum(labels == 1)
        recall = tp / actual_positives if actual_positives > 0 else 0.0
        review_rate = np.mean(preds)

        sweep_results.append({"threshold": float(t), "recall": float(recall), "review_rate": float(review_rate)})

        if recall >= 0.95:
            if review_rate < min_review_rate:
                min_review_rate = review_rate
                best_threshold = float(t)

    # If none reached 0.95 exactly, pick lowest threshold with max recall
    if min_review_rate == 1.0 and best_threshold == 0.5:
        best_threshold = float(thresholds[0])

    guardrail_decision = {
        "chosen_threshold": best_threshold,
        "achieved_recall_at_chosen": next(s["recall"] for s in sweep_results if s["threshold"] == best_threshold),
        "review_rate": min_review_rate,
        "sweep": sweep_results,
    }

    with open(out_dir / "guardrail_threshold.json", "w") as f:
        json.dump(guardrail_decision, f, indent=2)

    print(f"Guardrail threshold chosen: {best_threshold}, review rate: {min_review_rate:.4f}")


if __name__ == "__main__":
    main()
