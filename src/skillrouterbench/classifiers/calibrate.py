"""Temperature scaling calibration, ECE, and reliability diagrams."""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset


def load_jsonl(path: Path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def compute_ece_brier(logits, labels, n_bins=15):
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / len(confidences)) * np.abs(bin_acc - bin_conf)
            bin_stats.append({"bin_lower": bin_lower, "bin_upper": bin_upper, "accuracy": bin_acc, "confidence": bin_conf, "size": int(bin_size)})

    # Brier score (multi-class one-hot brier score)
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    return float(ece), float(brier), bin_stats


def main():
    parser = argparse.ArgumentParser(description="Calibrate encoder model with temperature scaling")
    parser.add_argument("--task", type=str, required=True, choices=["router", "guardrail"])
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--data_dir", type=str, default="data/cmg")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    model_path = Path(args.model_dir) / args.task
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    val_data = load_jsonl(Path(args.data_dir) / "val.jsonl")
    if args.task == "router":
        INTENTS = [
            "product_information", "adverse_event_report", "off_label_request",
            "access_and_reimbursement", "clinical_trial_inquiry",
            "medical_literature_request", "speaker_program_logistics"
        ]
        label2id = {l: i for i, l in enumerate(INTENTS)}
        labels = np.array([label2id[x["intent"]] for x in val_data])
    else:
        label2id = {"allow": 0, "needs_human_review": 1}
        labels = np.array([label2id[x["guardrail"]] for x in val_data])

    texts = [x["text"] for x in val_data]

    # Get uncalibrated logits
    all_logits = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            all_logits.append(outputs.logits.numpy())

    logits = np.vstack(all_logits)

    # Before calibration ECE & Brier
    ece_before, brier_before, _ = compute_ece_brier(logits, labels)

    # Fit temperature scaling
    scaler = TemperatureScaler()
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)
    loss_fn = nn.CrossEntropyLoss()

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    def eval_closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits_tensor)
        loss = loss_fn(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_closure)
    learned_temp = scaler.temperature.item()

    # After calibration logits
    calibrated_logits = (logits_tensor / learned_temp).detach().numpy()
    ece_after, brier_after, bin_stats = compute_ece_brier(calibrated_logits, labels)

    out_results_dir = Path(args.out_dir)
    out_results_dir.mkdir(parents=True, exist_ok=True)

    calib_data = {
        "task": args.task,
        "temperature": float(learned_temp),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "bin_stats": [
            {
                "bin_lower": float(b["bin_lower"]),
                "bin_upper": float(b["bin_upper"]),
                "accuracy": float(b["accuracy"]),
                "confidence": float(b["confidence"]),
                "size": int(b["size"]),
            }
            for b in bin_stats
        ],
    }

    with open(out_results_dir / f"calibration_{args.task}.json", "w") as f:
        json.dump(calib_data, f, indent=2)

    # Plot reliability diagram
    fig_dir = out_results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    confs = [b["confidence"] for b in bin_stats]
    accs = [b["accuracy"] for b in bin_stats]
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(confs, accs, marker="o", color="blue", label=f"Calibrated (T={learned_temp:.2f})")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram - {args.task.capitalize()} (ECE: {ece_before:.4f} -> {ece_after:.4f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_dir / f"calibration_{args.task}.png")
    plt.close()

    print(f"Calibration completed for {args.task}. Temp: {learned_temp:.4f}, ECE: {ece_before:.4f} -> {ece_after:.4f}")


if __name__ == "__main__":
    main()
