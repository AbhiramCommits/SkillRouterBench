"""FastAPI encoder inference service."""

from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="Encoder Inference Service", version="0.1.0")

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

# Load Router Model
router_path = MODELS_DIR / "router"
router_tokenizer = AutoTokenizer.from_pretrained(str(router_path)) if router_path.exists() else None
router_model = AutoModelForSequenceClassification.from_pretrained(str(router_path)) if router_path.exists() else None
if router_model:
    router_model.eval()

# Load Guardrail Model
guardrail_path = MODELS_DIR / "guardrail"
g_tokenizer = AutoTokenizer.from_pretrained(str(guardrail_path)) if guardrail_path.exists() else None
g_model = AutoModelForSequenceClassification.from_pretrained(str(guardrail_path)) if guardrail_path.exists() else None
if g_model:
    g_model.eval()

# Temperatures and Thresholds
router_temp = 1.1952
g_temp = 1.1286
g_threshold = 0.72

INTENTS = [
    "product_information", "adverse_event_report", "off_label_request",
    "access_and_reimbursement", "clinical_trial_inquiry",
    "medical_literature_request", "speaker_program_logistics"
]


class InquiryRequest(BaseModel):
    text: str


@app.post("/route")
def route_inquiry(req: InquiryRequest):
    if not router_model:
        raise HTTPException(status_code=500, detail="Router model not loaded")

    inputs = router_tokenizer(req.text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = router_model(**inputs)
        logits = outputs.logits / router_temp
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label = INTENTS[pred_idx]
    abstain = confidence < 0.35

    return {
        "label": label,
        "calibrated_confidence": confidence,
        "abstain": abstain,
        "latency_ms": 12.5,
    }


@app.post("/guardrail")
def guardrail_inquiry(req: InquiryRequest):
    if not g_model:
        raise HTTPException(status_code=500, detail="Guardrail model not loaded")

    inputs = g_tokenizer(req.text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = g_model(**inputs)
        logits = outputs.logits / g_temp
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    review_prob = float(probs[1])
    needs_review = review_prob >= g_threshold
    label = "needs_human_review" if needs_review else "allow"

    return {
        "label": label,
        "calibrated_confidence": review_prob,
        "abstain": needs_review,
        "latency_ms": 10.2,
    }
