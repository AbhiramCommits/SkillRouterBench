"""Agent orchestrator and configuration dataclass."""

import argparse
import json
import time
from dataclasses import dataclass

from skillrouterbench.agent.llm import AnthropicWrapper
from skillrouterbench.mcp_servers.docs_server import search_documents
from skillrouterbench.mcp_servers.tabular_server import query_metrics


@dataclass
class AgentConfig:
    instructions_level: str = "detailed"  # minimal or detailed
    skills_enabled: bool = True
    context_k: int = 3
    tools_enabled: str = "docs_plus_tabular"  # none, docs_only, docs_plus_tabular
    memory_enabled: bool = False


@dataclass
class AgentResult:
    inquiry_id: str
    intent: str
    guardrail: str
    escalated: bool
    cited_docs: list
    metrics_used: list
    answer: str
    token_usage: dict
    cost: float
    latency_ms: float


class AgentRuntime:
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.llm = AnthropicWrapper()

    def run(self, inquiry_id: str, text: str) -> AgentResult:
        start_time = time.time()

        # Step 1: Guardrail & Router simulation (using local encoder logic or rules)
        # For agent demo/runtime, let's use rule/classifier logic
        is_ae_or_offlabel = any(w in text.lower() for w in ["adverse", "anaphylaxis", "neutropenia", "off-label", "pediatric under", "unapproved"])
        guardrail_label = "needs_human_review" if is_ae_or_offlabel else "allow"

        if guardrail_label == "needs_human_review":
            latency = (time.time() - start_time) * 1000.0
            return AgentResult(
                inquiry_id=inquiry_id,
                intent="adverse_event_report" if "adverse" in text.lower() else "off_label_request",
                guardrail="needs_human_review",
                escalated=True,
                cited_docs=[],
                metrics_used=[],
                answer="Inquiry escalated to human medical review due to safety guardrail policy.",
                token_usage={"input_tokens": 0, "output_tokens": 0},
                cost=0.0,
                latency_ms=latency,
            )

        # Step 2: Intent routing
        intent = "product_information"
        if "reimbursement" in text.lower() or "billing" in text.lower() or "code" in text.lower():
            intent = "access_and_reimbursement"
        elif "trial" in text.lower() or "study" in text.lower():
            intent = "clinical_trial_inquiry"
        elif "literature" in text.lower() or "reprints" in text.lower():
            intent = "medical_literature_request"
        elif "speaker" in text.lower() or "symposium" in text.lower():
            intent = "speaker_program_logistics"

        # Step 3: Evidence Lookup via MCP tools
        cited_docs = []
        metrics_used = []
        retrieved_text = ""

        if self.config.tools_enabled in ["docs_only", "docs_plus_tabular"]:
            docs_json = search_documents(text, top_k=self.config.context_k)
            docs = json.loads(docs_json)
            for d in docs:
                cited_docs.append(d["doc_id"])
                retrieved_text += f"\n[{d['doc_id']}] {d['title']}: {d['text']}"

        if self.config.tools_enabled == "docs_plus_tabular":
            metrics_json = query_metrics(aggregate="mean_volume")
            m_data = json.loads(metrics_json)
            metrics_used.append(m_data)

        # Step 4: LLM Drafting
        system_prompt = "You are a professional pharma medical affairs AI assistant." if self.config.instructions_level == "detailed" else "Answer the inquiry."
        prompt = f"Inquiry: {text}\nIntent: {intent}\nRetrieved Evidence: {retrieved_text}\nProvide a structured answer with citations."

        llm_res = self.llm.call(prompt, system=system_prompt, fixture_name=f"agent_run_{intent}")

        latency = (time.time() - start_time) * 1000.0

        return AgentResult(
            inquiry_id=inquiry_id,
            intent=intent,
            guardrail="allow",
            escalated=False,
            cited_docs=cited_docs,
            metrics_used=metrics_used,
            answer=llm_res["content"],
            token_usage=llm_res["usage"],
            cost=llm_res["cost"],
            latency_ms=latency,
        )


def main():
    parser = argparse.ArgumentParser(description="Run Agent Runtime Demo")
    parser.add_argument("--demo", action="store_true", help="Run a demo inquiry")
    args = parser.parse_args()

    if args.demo:
        runtime = AgentRuntime(AgentConfig())
        res = runtime.run("CMG-99999", "Can you provide standard dosing guidelines for OncoShield?")
        print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()
