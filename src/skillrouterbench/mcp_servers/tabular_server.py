"""MCP tabular metrics query server using pandas."""

import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TabularServer")

METRICS_PATH = Path("data/kb/metrics.csv")
df_metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()


@mcp.tool()
def query_metrics(region: str = None, quarter: str = None, product: str = None, aggregate: str = None) -> str:
    """Query fictional commercial metrics dataset with optional filtering and aggregation."""
    if df_metrics.empty:
        return json.dumps([])

    df = df_metrics.copy()
    if region:
        df = df[df["region"].str.lower() == region.lower()]
    if quarter:
        df = df[df["quarter"].str.lower() == quarter.lower()]
    if product:
        df = df[df["product"].str.lower() == product.lower()]

    if aggregate == "mean_volume":
        result = {"mean_inquiry_volume": float(df["inquiry_volume"].mean())}
        return json.dumps(result)
    elif aggregate == "mean_response_hours":
        result = {"mean_response_hours": float(df["median_response_hours"].mean())}
        return json.dumps(result)
    elif aggregate == "mean_escalation_rate":
        result = {"mean_escalation_rate": float(df["escalation_rate"].mean())}
        return json.dumps(result)

    records = df.to_dict(orient="records")
    return json.dumps(records)


if __name__ == "__main__":
    mcp.run()
