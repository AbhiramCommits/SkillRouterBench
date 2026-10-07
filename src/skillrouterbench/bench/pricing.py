"""Pricing rates table for API models."""

PRICING = {
    "self_hosted_encoder": {"input_per_1k": 0.0001, "output_per_1k": 0.0001},
    "hosted_llm_zero_shot": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "hosted_llm_few_shot": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "tfidf_logreg_floor": {"input_per_1k": 0.0, "output_per_1k": 0.0},
}
