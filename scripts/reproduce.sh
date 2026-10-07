#!/usr/bin/env bash
set -e

echo "=== SkillRouterBench End-to-End Reproduction Script ==="
echo "1. Generating synthetic dataset and knowledge base..."
python -m skillrouterbench.data.generate --out data/

echo "2. Fine-tuning router encoder..."
python -m skillrouterbench.classifiers.train_encoder --task router --epochs 1

echo "3. Fine-tuning guardrail encoder..."
python -m skillrouterbench.classifiers.train_encoder --task guardrail --epochs 1

echo "4. Running temperature scaling calibration..."
python -m skillrouterbench.classifiers.calibrate --task router
python -m skillrouterbench.classifiers.calibrate --task guardrail

echo "5. Optimizing guardrail threshold..."
python -m skillrouterbench.classifiers.threshold

echo "6. Running head-to-head component benchmark..."
python -m skillrouterbench.bench.classifier_bench

echo "7. Building component decision table..."
python -m skillrouterbench.bench.decision_table

echo "8. Running failure modes analysis..."
python -m skillrouterbench.analysis.failure_modes

echo "9. Running agent ablation study..."
python -m skillrouterbench.bench.agent_ablation

echo "10. Running test suite..."
pytest -v

echo "=== Reproduction Complete Successfully! ==="
