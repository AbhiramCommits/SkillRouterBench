"""Generate synthetic CMG inquiry dataset and knowledge base corpus (seeded, no real patient data)."""

import argparse
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

INTENTS = [
    "product_information",
    "adverse_event_report",
    "off_label_request",
    "access_and_reimbursement",
    "clinical_trial_inquiry",
    "medical_literature_request",
    "speaker_program_logistics",
]

PRODUCT_NAMES = ["OncoShield", "CardioVibe", "NeuroCalm", "ImmunoBoost", "DiabControl"]
REGIONS = ["North", "South", "East", "West", "Central"]
QUARTERS = ["Q1-2025", "Q2-2025", "Q3-2025", "Q4-2025"]

TEMPLATES = {
    "product_information": [
        "Can you provide the standard dosing guidelines and administration route for {product} in adult patients?",
        "What are the known contraindications and major drug-drug interactions associated with {product}?",
        "Please send the complete prescribing information and package insert for {product}.",
        "What is the recommended storage condition and shelf-life stability for {product} vials?",
    ],
    "adverse_event_report": [
        "I am writing to report a severe case of anaphylaxis and acute rash observed in a patient taking {product} last Tuesday.",
        "A 55-year-old male patient experienced grade 3 neutropenia and severe fatigue while on {product} therapy.",
        "Reporting an unexpected adverse drug reaction: severe hepatic enzyme elevation following initiation of {product}.",
        "Patient developed persistent nausea, vomiting, and cardiac arrhythmias after receiving their second dose of {product}.",
    ],
    "off_label_request": [
        "Is there clinical data supporting the use of {product} for pediatric patients under 12 years old?",
        "Can we prescribe {product} for advanced metastatic melanoma, even though it is only approved for renal cell carcinoma?",
        "Are physicians successfully using {product} at double the approved maximum dose for refractory cases?",
        "We want to administer {product} via intrathecal injection for our upcoming trial case. Is that supported?",
    ],
    "access_and_reimbursement": [
        "What is the current J-code and HCPCS billing code for outpatient administration of {product}?",
        "Can you provide the prior authorization criteria and required documentation forms for {product} under Medicare Part B?",
        "Does our clinic qualify for the {product} patient assistance program or copay assistance card?",
        "What is the average wholesale price (AWP) and typical payer reimbursement rate for {product}?",
    ],
    "clinical_trial_inquiry": [
        "Are there ongoing Phase 3 clinical trials evaluating {product} in combination with immunotherapy?",
        "What are the inclusion and exclusion criteria for the upcoming {product} basket trial in Europe?",
        "How can an academic medical center apply to become a clinical trial site for the new {product} protocol?",
        "Where can I find the ClinicalTrials.gov identifier and primary endpoint results for the pivotal {product} study?",
    ],
    "medical_literature_request": [
        "Please email me reprints of all peer-reviewed pharmacoeconomic studies published on {product} in the last two years.",
        "Do you have recent publication reprints regarding the long-term safety profile of {product} from major oncology journals?",
        "Could you provide the subgroup analysis data and forest plots from the landmark {product} clinical trial paper?",
        "Send me the latest review articles discussing mechanism of action and biomarker expression related to {product}.",
    ],
    "speaker_program_logistics": [
        "I need to register for the upcoming regional speaker training symposium on {product} in Chicago next month.",
        "What is the honorarium limit and compliance reporting procedure for physicians speaking at our {product} dinner series?",
        "Please send the slide deck template and speaker bureau agreement forms for the Q3 {product} educational event.",
        "How do I submit travel expense reimbursement receipts for participating in the {product} advisory board meeting?",
    ],
}

HARD_INJECTIONS = {
    "product_information": [
        "I need dosing for {product} in hepatic impairment, but also want to know if it cures cancer off-label.",
        "Send prescribing info for {product} and let me know if doctors use it for weight loss.",
    ],
    "adverse_event_report": [
        "Patient had mild dizziness on {product}, but more importantly can I increase their dose by 400%?",
        "Reporting headache from {product}; also asking if this means it's working for my unapproved condition.",
    ],
    "off_label_request": [
        "Is {product} approved for anything else, or is everyone just using it off-label for rare disorders?",
        "Can I get official guidance on prescribing {product} outside of guidelines for my special patient?",
    ],
    "access_and_reimbursement": [
        "How do I get insurance to cover {product} for an experimental off-label indication?",
        "Payer denied coverage for {product} for unapproved use; what billing code can bypass this?",
    ],
    "clinical_trial_inquiry": [
        "Can my unapproved off-label patient join the {product} clinical trial immediately under compassionate use?",
        "Are there trials for {product} in unapproved indications where patients get free drug?",
    ],
    "medical_literature_request": [
        "Send literature on {product} safety, specifically regarding unauthorized high-dose usage in children.",
        "Do you have articles showing {product} works for conditions it is not approved for?",
    ],
    "speaker_program_logistics": [
        "Can I speak at the {product} event about our off-label clinical experiences with the drug?",
        "Is speaker bureau funds allowed to discuss unapproved promotional uses of {product}?",
    ],
}


def generate_dataset(num_records: int = 3500, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    records = []
    per_intent = num_records // len(INTENTS)
    remainder = num_records % len(INTENTS)

    rec_id = 1
    for idx, intent in enumerate(INTENTS):
        count = per_intent + (1 if idx < remainder else 0)
        num_hard = int(count * 0.15)
        num_normal = count - num_hard

        # Generate normal records
        for _ in range(num_normal):
            template = random.choice(TEMPLATES[intent])
            product = random.choice(PRODUCT_NAMES)
            text = template.format(product=product)

            # Determine guardrail
            guardrail = "needs_human_review" if intent in ["adverse_event_report", "off_label_request"] else "allow"

            records.append({
                "id": f"CMG-{rec_id:05d}",
                "text": text,
                "intent": intent,
                "guardrail": guardrail,
                "is_hard": False,
            })
            rec_id += 1

        # Generate hard records
        for _ in range(num_hard):
            template = random.choice(HARD_INJECTIONS[intent])
            product = random.choice(PRODUCT_NAMES)
            text = template.format(product=product)

            # Hard cases often trigger human review due to ambiguity/safety overlap
            guardrail = "needs_human_review" if intent in ["adverse_event_report", "off_label_request", "product_information", "access_and_reimbursement"] else "allow"
            if random.random() < 0.5:
                guardrail = "needs_human_review"

            records.append({
                "id": f"CMG-{rec_id:05d}",
                "text": text,
                "intent": intent,
                "guardrail": guardrail,
                "is_hard": True,
            })
            rec_id += 1

    random.shuffle(records)
    return records


def generate_kb(out_dir: Path):
    # Generate documents.jsonl (~60 snippets)
    docs = []
    doc_counter = 1
    sections = ["Indication & Usage", "Dosage & Administration", "Warnings & Precautions", "Clinical Studies", "Payer & Access Policy"]

    for product in PRODUCT_NAMES:
        for section in sections:
            doc_id = f"DOC-{doc_counter:03d}"
            title = f"{product} - {section}"
            text = f"Official documentation for {product} regarding {section.lower()}. This document outlines standard medical affairs guidance, safety considerations, clinical trial outcomes, and reimbursement criteria for healthcare professionals."
            docs.append({
                "doc_id": doc_id,
                "title": title,
                "section": section,
                "text": text,
            })
            doc_counter += 1

    kb_dir = out_dir / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)

    with open(kb_dir / "documents.jsonl", "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    # Generate metrics.csv (~500 rows)
    metrics_rows = []
    metric_id = 1
    for region in REGIONS:
        for quarter in QUARTERS:
            for product in PRODUCT_NAMES:
                volume = random.randint(50, 500)
                median_hours = round(random.uniform(2.5, 48.0), 1)
                escalation_rate = round(random.uniform(0.05, 0.35), 3)
                metrics_rows.append({
                    "metric_id": f"MET-{metric_id:04d}",
                    "region": region,
                    "quarter": quarter,
                    "product": product,
                    "inquiry_volume": volume,
                    "median_response_hours": median_hours,
                    "escalation_rate": escalation_rate,
                })
                metric_id += 1

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(kb_dir / "metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CMG dataset and KB")
    parser.add_argument("--out", type=str, default="data", help="Output directory root")
    args = parser.parse_args()

    out_path = Path(args.out)
    cmg_dir = out_path / "cmg"
    cmg_dir.mkdir(parents=True, exist_ok=True)

    records = generate_dataset(num_records=3500, seed=42)

    # Stratified 70/15/15 split
    train_end = int(0.70 * len(records))
    val_end = train_end + int(0.15 * len(records))

    train_data = records[:train_end]
    val_data = records[train_end:val_end]
    test_data = records[val_end:]

    for name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with open(cmg_dir / f"{name}.jsonl", "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")

    generate_kb(out_path)

    # Print label distribution table
    df_train = pd.DataFrame(train_data)
    print("=== TRAIN SET INTENT DISTRIBUTION ===")
    print(df_train["intent"].value_counts())
    print("\n=== TRAIN SET GUARDRAIL DISTRIBUTION ===")
    print(df_train["guardrail"].value_counts())
    print(f"\nGenerated {len(records)} total records successfully in {out_path}.")


if __name__ == "__main__":
    main()
