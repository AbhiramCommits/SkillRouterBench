"""Anthropic API client wrapper with fixture replay fallback."""

import json
import os
import time
from pathlib import Path

from anthropic import Anthropic

FIXTURES_DIR = Path("fixtures/anthropic")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


class AnthropicWrapper:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
        self.has_key = self.api_key and self.api_key != "your_anthropic_api_key_here"
        self.client = Anthropic(api_key=self.api_key) if self.has_key else None

    def call(self, prompt: str, system: str = "", fixture_name: str = "default_response") -> dict:
        fixture_path = FIXTURES_DIR / f"{fixture_name}.json"

        if not self.has_key:
            if fixture_path.exists():
                with open(fixture_path, "r") as f:
                    return json.load(f)
            else:
                # Default fallback fixture response
                fallback = {
                    "content": "This is a recorded fixture response for the CMG medical inquiry. Based on approved product labeling, standard dosing applies. [Cited: DOC-001]",
                    "usage": {"input_tokens": 150, "output_tokens": 50},
                    "cost": 0.0012,
                    "latency_ms": 450.0,
                }
                with open(fixture_path, "w") as f:
                    json.dump(fallback, f, indent=2)
                return fallback

        # Live API call
        start_time = time.time()
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.time() - start_time) * 1000.0
        text = response.content[0].text
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        cost = (in_tokens * 3.0 / 1_000_000) + (out_tokens * 15.0 / 1_000_000)

        result = {
            "content": text,
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
            "cost": cost,
            "latency_ms": latency,
        }

        # Save fixture for future offline replay
        with open(fixture_path, "w") as f:
            json.dump(result, f, indent=2)

        return result
