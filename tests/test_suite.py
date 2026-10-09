"""Comprehensive unit and integration test suite."""

import json
import os

import numpy as np

from skillrouterbench.agent.llm import AnthropicWrapper
from skillrouterbench.agent.runtime import AgentConfig, AgentRuntime
from skillrouterbench.bench.classifier_bench import bootstrap_ci
from skillrouterbench.bench.pricing import PRICING
from skillrouterbench.classifiers.calibrate import compute_ece_brier
from skillrouterbench.data.generate import generate_dataset
from skillrouterbench.mcp_servers.docs_server import get_document, search_documents
from skillrouterbench.mcp_servers.tabular_server import query_metrics


def test_dataset_generation_determinism():
    records1 = generate_dataset(num_records=100, seed=42)
    records2 = generate_dataset(num_records=100, seed=42)
    assert len(records1) == 100
    assert records1[0]["id"] == records2[0]["id"]
    assert records1[0]["text"] == records2[0]["text"]
    assert records1[0]["intent"] == records2[0]["intent"]


def test_ece_brier_computation():
    logits = np.array([[2.0, 0.5], [1.5, 1.2], [0.1, 2.5]])
    labels = np.array([0, 0, 1])
    ece, brier, bin_stats = compute_ece_brier(logits, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 2.0
    assert isinstance(bin_stats, list)


def test_mcp_docs_server():
    res_json = search_documents("OncoShield dosing", top_k=2)
    docs = json.loads(res_json)
    assert isinstance(docs, list)
    if len(docs) > 0:
        assert "doc_id" in docs[0]
        assert "text" in docs[0]
        doc_id = docs[0]["doc_id"]
        single_doc = json.loads(get_document(doc_id))
        assert single_doc["doc_id"] == doc_id


def test_mcp_tabular_server():
    res_json = query_metrics(region="North", aggregate="mean_volume")
    data = json.loads(res_json)
    assert "mean_inquiry_volume" in data
    assert isinstance(data["mean_inquiry_volume"], float)


def test_anthropic_wrapper_offline_fallback():
    os_key_backup = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key_here"
    wrapper = AnthropicWrapper()
    res = wrapper.call("Test prompt", fixture_name="test_fixture")
    assert "content" in res
    assert "usage" in res
    assert "cost" in res
    if os_key_backup:
        os.environ["ANTHROPIC_API_KEY"] = os_key_backup
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)


def test_agent_runtime_configs():
    config = AgentConfig(instructions_level="detailed", skills_enabled=True, context_k=2, tools_enabled="docs_only")
    runtime = AgentRuntime(config)
    res = runtime.run("TEST-001", "What are the prescribing guidelines for CardioVibe?")
    assert res.inquiry_id == "TEST-001"
    assert res.intent in ["product_information", "access_and_reimbursement", "clinical_trial_inquiry", "medical_literature_request", "speaker_program_logistics"]
    assert isinstance(res.answer, str)


def test_pricing_math():
    assert PRICING["self_hosted_encoder"]["input_per_1k"] == 0.0001
    assert PRICING["hosted_llm_zero_shot"]["output_per_1k"] == 0.015


def test_bootstrap_ci_bounds():
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1, 0, 1, 2]
    low, high = bootstrap_ci(y_true, y_pred, lambda a, b, average, zero_division: 0.85, num_samples=50)
    assert low <= high
