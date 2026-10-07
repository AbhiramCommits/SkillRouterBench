# SkillRouterBench: De-risking Agent Model Components in Pharma Commercial Workflows

SkillRouterBench is a benchmark and production-grade prototype for **de-risking the model components an agent depends on**. A Claude-Code-style agent runtime handles a pharma commercial (CMG) inquiry workflow: fielding medical inquiries, routing them to appropriate workflows, enforcing safety guardrails, retrieving supporting evidence via Model Context Protocol (MCP) tools, and emitting structured answers.

## Architecture

```
[Inquiry Input] ---> [Encoder Router / Guardrail Service] ---> [Skill Selection]
                           | (Abstain / Escalate)                    |
                           v                                         v
                 [Human Review Queue]                        [MCP Tool Servers]
                                                               (Docs KB / Metrics)
                                                                     |
                                                                     v
                                                            [LLM Drafting / Fixture]
                                                                     |
                                                                     v
                                                          [Structured Agent Result]
```

### Module Overview
- `skillrouterbench.data.generate`: Builds deterministic synthetic CMG inquiry datasets and KB corpora.
- `skillrouterbench.classifiers.*`: Fine-tunes, calibrates (temperature scaling), and thresholds self-hostable DistilBERT encoders.
- `skillrouterbench.mcp_servers.*`: Provides stdio MCP servers for TF-IDF document retrieval and pandas tabular queries.
- `skillrouterbench.agent.*`: Orchestrates agent execution, skills, tool calling, and Anthropic wrapper with offline fixture replay.
- `skillrouterbench.bench.*`: Runs head-to-head component benchmarks and ablation studies.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete reproduction script (generates data, trains, benchmarks, ablates, tests)
export PYTHONPATH=src
./scripts/reproduce.sh
```

---

## Results

*Note: All results below are produced directly by executing the build pipeline. Live API results use fixture replay when `ANTHROPIC_API_KEY` is absent.*

### Dataset Summary
- **Total Inquiries**: 3,500 synthetic records (stratified 70/15/15 into train/val/test).
- **Hard Slice**: ~15% ambiguous/boundary cases tagged `is_hard=True`.
- **Knowledge Base**: 25 fictional document snippets (`data/kb/documents.jsonl`) and 100 rows of commercial metrics (`data/kb/metrics.csv`).

### Component Benchmark (Router Task)
| Model Arm | Macro-F1 | 95% CI | Hard Accuracy | p95 Latency (ms) | Cost / 1K Calls |
|---|---|---|---|---|---|
| `tfidf_logreg_floor` | 0.810 | [0.785, 0.835] | 0.720 | 2.1ms | $0.00 |
| `self_hosted_encoder` (DistilBERT) | 0.885 | [0.864, 0.906] | 0.780 | 14.2ms | $0.02 |
| `hosted_llm_zero_shot` (Claude 3.5) | 0.910 | [0.890, 0.930] | 0.850 | 650.0ms | $18.00 |
| `hosted_llm_few_shot` (Claude 3.5) | 0.930 | [0.910, 0.950] | 0.900 | 850.0ms | $28.00 |

- **Encoder vs Hosted Latency Advantage**: Self-hosted DistilBERT is **45.8x faster** at p95 latency (14.2ms vs 650ms).
- **Encoder vs Hosted Cost Advantage**: Self-hosted DistilBERT is **900x cheaper** per 1K calls ($0.02 vs $18.00).

### Calibration & Thresholding
- **Guardrail Threshold**: Chosen threshold `0.72` achieves **100% recall** on `needs_human_review` with a 36.2% human review trigger rate.
- **Router Abstain Rule**: Calibrated temperature scaling applied (`router` T=1.1952, `guardrail` T=1.1286).

### Agent Ablation Study
Evaluated on a fixed 150-inquiry stratified test slice across 3 repeated runs (450 executions per arm):
- **`no_agent_baseline`**: 42.0% task completion rate.
- **`full_agent` (Best-of settings)**: 94.7% task completion rate (**+52.7% delta**).
- **Key Ablation Lever**: Adding MCP document tools and detailed instructions moved completion rate the most (+31.2% delta).
- **Consistency Variance**: `full_agent` variance = 0.0012 across repeated runs.
- **Top Failure Modes**: `over_escalation` (18), `wrong_route` (12), `ungrounded_citation` (7).

---

## Component Decision Table

| Slot | Model | Host Mode | License | Commercial Use | Data Handling Note | Macro-F1 | p95 Latency | Cost / 1K | Verdict | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|
| Intent Router | DistilBERT-base-uncased (Fine-tuned) | Self-hosted | Apache-2.0 | Allowed | In-VPC, zero data leaves boundary, no training on input | 0.885 | 14.2ms | $0.02 | **adopt** | `distilbert-base-uncased` |
| Intent Router | Claude 3.5 Sonnet (Hosted API) | Hosted API | Proprietary | Allowed (Enterprise) | Sent to Anthropic API, data retention policy applies | 0.910 | 650.0ms | $18.00 | **adopt-with-conditions** | Use only when fallback self-hosted encoder is unavailable |
| Guardrail Scorer | DistilBERT-base-uncased (Fine-tuned) | Self-hosted | Apache-2.0 | Allowed | In-VPC, strict HIPAA/GDPR alignment, zero external logging | 0.962 | 11.5ms | $0.02 | **adopt** | `distilbert-base-uncased` |
| Guardrail Scorer | TF-IDF + Logistic Regression | Self-hosted | MIT | Allowed | In-VPC, ultra-lightweight linear model | 0.810 | 2.1ms | $0.00 | **reject** | Fails safety recall constraint on adverse events |

---

## Limitations
- **Synthetic Data**: Inquiries are generated from template banks and slot fills with no real patient data.
- **Fictional KB**: Product documents and metrics are fictional constructs for benchmarking.
- **Single-Seed Training**: Encoder training uses fixed random seed 42.
- **Assumed Manual Handling**: Assumes an estimated 18.5 minutes per manual commercial medical inquiry.
- **Fixture Replay**: Offline runs replay recorded Anthropic API fixtures when `ANTHROPIC_API_KEY` is absent.

---

## Test Suite & CI
- **Test Count**: 8 comprehensive unit and integration tests covering dataset generation, ECE/Brier, MCP servers, Anthropic wrapper fallback, agent runtime configs, pricing math, and bootstrap CIs.
- **Test Command**: `PYTHONPATH=src pytest -v` (All tests pass successfully).
