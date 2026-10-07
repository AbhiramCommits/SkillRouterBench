# Triage Inquiry Skill

When to use: At the start of every commercial medical inquiry to categorize intent and check safety guardrails.

Procedure:
1. Receive the raw field-medical inquiry text.
2. Query the routing classifier to determine the primary intent category.
3. Query the guardrail classifier to check if safety escalation is required.
4. If safety score exceeds threshold or intent demands escalation, initiate handoff.
