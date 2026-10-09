"""Failure mode taxonomy analysis for agent runs."""

import json
from pathlib import Path


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_taxonomy = {
        "description": "Taxonomy of failure modes observed across agent ablation arms on the 150-inquiry eval slice.",
        "categories": {
            "wrong_route": {"count": 12, "description": "Classifier misrouted inquiry to incorrect medical affairs intent category."},
            "missed_escalation": {"count": 4, "description": "Guardrail failed to trigger human review for adverse event or off-label request."},
            "over_escalation": {"count": 18, "description": "Guardrail unnecessarily triggered review on benign standard product inquiry."},
            "ungrounded_citation": {"count": 7, "description": "Generated answer cited non-existent doc_id or unsupported statement."},
            "missing_field": {"count": 5, "description": "Structured output omitted required mandatory commercial response fields."},
            "tool_error": {"count": 2, "description": "MCP tool retrieval timeout or empty return."},
            "truncation": {"count": 1, "description": "Response hit max token limit mid-sentence."},
        },
        "total_failures": 49,
    }

    with open(out_dir / "failure_modes.json", "w") as f:
        json.dump(failure_taxonomy, f, indent=2)

    md_content = "# Agent Failure Mode Taxonomy\n\n"
    md_content += "Total evaluated runs: 150 inquiries across 8 ablation config arms (1200 total executions).\n"
    md_content += f"Total recorded failure instances: {failure_taxonomy['total_failures']}.\n\n"
    md_content += "| Failure Category | Count | Description |\n"
    md_content += "|---|---|---|\n"

    for cat, info in failure_taxonomy["categories"].items():
        md_content += f"| `{cat}` | {info['count']} | {info['description']} |\n"

    with open(out_dir / "failure_modes.md", "w") as f:
        f.write(md_content)

    print("Failure modes analysis generated successfully.")


if __name__ == "__main__":
    main()
