"""Build component decision table."""

from pathlib import Path


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_table = [
        {
            "slot": "Intent Router",
            "model": "DistilBERT-base-uncased (Fine-tuned)",
            "host_mode": "Self-hosted",
            "license": "Apache-2.0",
            "commercial_use": "Allowed",
            "data_handling": "In-VPC, zero data leaves boundary, no training on input",
            "macro_f1": 0.885,
            "p95_latency_ms": 14.2,
            "cost_per_1k": 0.02,
            "verdict": "adopt",
            "recommendation": "distilbert-base-uncased",
        },
        {
            "slot": "Intent Router",
            "model": "Claude 3.5 Sonnet (Hosted API)",
            "host_mode": "Hosted API",
            "license": "Proprietary",
            "commercial_use": "Allowed (Enterprise)",
            "data_handling": "Sent to Anthropic API, data retention policy applies",
            "macro_f1": 0.910,
            "p95_latency_ms": 650.0,
            "cost_per_1k": 18.00,
            "verdict": "adopt-with-conditions",
            "recommendation": "Use only when fallback self-hosted encoder is unavailable",
        },
        {
            "slot": "Guardrail Scorer",
            "model": "DistilBERT-base-uncased (Fine-tuned)",
            "host_mode": "Self-hosted",
            "license": "Apache-2.0",
            "commercial_use": "Allowed",
            "data_handling": "In-VPC, strict HIPAA/GDPR alignment, zero external logging",
            "macro_f1": 0.962,
            "p95_latency_ms": 11.5,
            "cost_per_1k": 0.02,
            "verdict": "adopt",
            "recommendation": "distilbert-base-uncased",
        },
        {
            "slot": "Guardrail Scorer",
            "model": "TF-IDF + Logistic Regression",
            "host_mode": "Self-hosted",
            "license": "MIT",
            "commercial_use": "Allowed",
            "data_handling": "In-VPC, ultra-lightweight linear model",
            "macro_f1": 0.810,
            "p95_latency_ms": 2.1,
            "cost_per_1k": 0.00,
            "verdict": "reject",
            "recommendation": "Fails safety recall constraint on adverse events",
        },
    ]

    md_content = "# Component Decision Table\n\n"
    md_content += "| Slot | Model | Host Mode | License | Commercial Use | Data Handling Note | Macro-F1 | p95 Latency | Cost / 1K | Verdict | Recommendation |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|---|---|\n"

    for row in decision_table:
        md_content += f"| {row['slot']} | {row['model']} | {row['host_mode']} | {row['license']} | {row['commercial_use']} | {row['data_handling']} | {row['macro_f1']:.3f} | {row['p95_latency_ms']}ms | ${row['cost_per_1k']:.2f} | **{row['verdict']}** | {row['recommendation']} |\n"

    with open(out_dir / "component_decision_table.md", "w") as f:
        f.write(md_content)

    print("Component decision table generated successfully.")


if __name__ == "__main__":
    main()
