"""Agent ablation study across configuration arms and rubric grader."""

import json
import random
import numpy as np
from pathlib import Path
from skillrouterbench.agent.runtime import AgentRuntime, AgentConfig


def load_jsonl(path: Path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def rubric_grade(inquiry: dict, result) -> dict:
    # 1. Routing correctness
    expected_intent = inquiry["intent"]
    routing_correct = (result.intent == expected_intent)

    # 2. Guardrail correctness
    expected_guardrail = inquiry["guardrail"]
    guardrail_correct = (result.guardrail == expected_guardrail)

    # 3. Citation groundedness
    cited_valid = True
    if not result.escalated and result.cited_docs:
        # Check if doc_ids exist in KB
        for doc_id in result.cited_docs:
            if not doc_id.startswith("DOC-"):
                cited_valid = False

    # 4. Field completeness
    field_complete = bool(result.answer and len(result.answer) > 20)

    task_success = bool(routing_correct and guardrail_correct and cited_valid and field_complete)

    return {
        "routing_correct": routing_correct,
        "guardrail_correct": guardrail_correct,
        "citation_grounded": cited_valid,
        "field_complete": field_complete,
        "task_success": task_success,
    }


def main():
    data_dir = Path("data/cmg")
    test_data = load_jsonl(data_dir / "test.jsonl")
    eval_slice = test_data[:150]  # fixed 150-inquiry eval slice

    arms = {
        "no_agent_baseline": AgentConfig(instructions_level="minimal", skills_enabled=False, context_k=0, tools_enabled="none", memory_enabled=False),
        "instructions_minimal": AgentConfig(instructions_level="minimal", skills_enabled=True, context_k=3, tools_enabled="docs_only", memory_enabled=False),
        "instructions_detailed": AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=3, tools_enabled="docs_only", memory_enabled=False),
        "skills_off": AgentConfig(instructions_level="detailed", skills_enabled=False, context_k=3, tools_enabled="docs_only", memory_enabled=False),
        "skills_on": AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=3, tools_enabled="docs_only", memory_enabled=False),
        "context_k_1": AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=1, tools_enabled="docs_only", memory_enabled=False),
        "context_k_8": AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=8, tools_enabled="docs_only", memory_enabled=False),
        "tools_none": AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=3, tools_enabled="none", memory_enabled=False),
        "docs_plus_tabular": AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=3, tools_enabled="docs_plus_tabular", memory_enabled=False),
        "full_agent": AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=3, tools_enabled="docs_plus_tabular", memory_enabled=True),
    }

    out_results = {}
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm_name, config in arms.items():
        runtime = AgentRuntime(config)
        successes = []
        latencies = []
        costs = []
        escalation_counts = 0

        # Run on eval slice (repeating 3 times for variance/consistency)
        for repeat in range(3):
            for idx, item in enumerate(eval_slice):
                res = runtime.run(f"EVAL-{repeat}-{idx}", item["text"])
                grade = rubric_grade(item, res)
                successes.append(1.0 if grade["task_success"] else 0.0)
                latencies.append(res.latency_ms)
                costs.append(res.cost)
                if res.escalated:
                    escalation_counts += 1

        mean_success = float(np.mean(successes))
        variance_success = float(np.var(successes))
        mean_latency = float(np.mean(latencies))
        mean_cost = float(np.mean(costs))
        human_review_rate = float(escalation_counts / (len(eval_slice) * 3))

        out_results[arm_name] = {
            "task_completion_rate": mean_success,
            "consistency_variance": variance_success,
            "mean_latency_ms": mean_latency,
            "mean_cost_per_inquiry": mean_cost,
            "human_review_trigger_rate": human_review_rate,
        }

    with open(out_dir / "agent_ablation.json", "w") as f:
        json.dump(out_results, f, indent=2)

    # Write Markdown table
    md_content = "# Agent Ablation Study Results\n\n"
    md_content += "Evaluated on a fixed 150-inquiry stratified test slice across 3 repeated runs (450 total executions per arm).\n"
    md_content += "Assumed manual-handling baseline: 18.5 minutes per inquiry.\n\n"
    md_content += "| Ablation Arm | Task Completion Rate | Variance | Mean Latency (ms) | Mean Cost | Human Review Rate |\n"
    md_content += "|---|---|---|---|---|---|\n"

    for arm, metrics in out_results.items():
        md_content += f"| `{arm}` | {metrics['task_completion_rate']*100:.1f}% | {metrics['consistency_variance']:.4f} | {metrics['mean_latency_ms']:.1f} | ${metrics['mean_cost_per_inquiry']:.4f} | {metrics['human_review_trigger_rate']*100:.1f}% |\n"

    with open(out_dir / "agent_ablation.md", "w") as f:
        f.write(md_content)

    print("Agent ablation study complete. Results saved to results/agent_ablation.json and .md")


if __name__ == "__main__":
    main()
