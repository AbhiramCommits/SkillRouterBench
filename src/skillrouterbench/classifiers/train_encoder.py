"""Fine-tune a HF sequence-classification model on CMG dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

INTENTS = [
    "product_information",
    "adverse_event_report",
    "off_label_request",
    "access_and_reimbursement",
    "clinical_trial_inquiry",
    "medical_literature_request",
    "speaker_program_logistics",
]


def load_jsonl(path: Path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune encoder for router or guardrail")
    parser.add_argument("--task", type=str, required=True, choices=["router", "guardrail"])
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--data_dir", type=str, default="data/cmg")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data_dir)
    train_raw = load_jsonl(data_path / "train.jsonl")
    val_raw = load_jsonl(data_path / "val.jsonl")

    if args.task == "router":
        label_list = INTENTS
        label2id = {lbl: i for i, lbl in enumerate(label_list)}
        id2label = {i: lbl for i, lbl in enumerate(label_list)}

        def process_item(item):
            return {"text": item["text"], "label": label2id[item["intent"]]}
    else:
        label_list = ["allow", "needs_human_review"]
        label2id = {lbl: i for i, lbl in enumerate(label_list)}
        id2label = {i: lbl for i, lbl in enumerate(label_list)}

        def process_item(item):
            return {"text": item["text"], "label": label2id[item["guardrail"]]}

    train_data = [process_item(x) for x in train_raw]
    val_data = [process_item(x) for x in val_raw]

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    output_model_dir = Path(args.out_dir) / args.task
    output_model_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_model_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        seed=args.seed,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
    )

    trainer.train()

    # Save model and config
    trainer.save_model(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))

    # Save training history log
    history = [
        {"epoch": log.get("epoch"), "train_loss": log.get("loss"), "eval_loss": log.get("eval_loss")}
        for log in trainer.state.log_history
        if "loss" in log or "eval_loss" in log
    ]
    with open(output_model_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Successfully trained {args.task} model. Saved to {output_model_dir}")


if __name__ == "__main__":
    main()
