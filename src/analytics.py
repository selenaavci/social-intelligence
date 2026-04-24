"""Core analytics: sentiment, engagement, reach, multi-bank benchmarking."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    if df.empty or "Sentiment" not in df.columns:
        return pd.Series(dtype=float)
    dist = df["Sentiment"].value_counts(normalize=True)
    for label in ("Positive", "Neutral", "Negative"):
        if label not in dist.index:
            dist[label] = 0.0
    return dist.reindex(["Positive", "Neutral", "Negative"]).fillna(0.0)


def engagement_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"posts": 0, "total_interaction": 0, "avg_interaction": 0.0,
                "total_reach": 0, "avg_engagement": 0.0}
    total_interaction = float(df.get("Interaction", pd.Series([0])).sum())
    total_reach = float(df.get("Views", pd.Series([0])).sum())
    avg_engagement = float(df.get("Engagement", pd.Series([0.0])).mean())
    return {
        "posts": int(len(df)),
        "total_interaction": total_interaction,
        "avg_interaction": total_interaction / max(len(df), 1),
        "total_reach": total_reach,
        "avg_engagement": avg_engagement,
    }


def source_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Source" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("Source").agg(
        posts=("Text", "size"),
        interaction=("Interaction", "sum"),
        engagement=("Engagement", "mean"),
        neg=("Sentiment", lambda s: (s == "Negative").mean()),
        pos=("Sentiment", lambda s: (s == "Positive").mean()),
    ).reset_index().sort_values("posts", ascending=False)
    return g


def daily_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=["Date"]).copy()
    d["day"] = d["Date"].dt.floor("D")
    daily = d.groupby("day").agg(
        posts=("Text", "size"),
        negative=("Sentiment", lambda s: (s == "Negative").sum()),
        positive=("Sentiment", lambda s: (s == "Positive").sum()),
        neutral=("Sentiment", lambda s: (s == "Neutral").sum()),
        interaction=("Interaction", "sum"),
    ).reset_index()
    daily["negative_share"] = daily["negative"] / daily["posts"].replace(0, np.nan)
    return daily.fillna(0.0)


def benchmark_banks(combined: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return summaries.copy()

    agg = combined.groupby("Bank").agg(
        posts=("Text", "size"),
        avg_engagement=("Engagement", "mean"),
        total_interaction=("Interaction", "sum"),
        neg_share_posts=("Sentiment", lambda s: (s == "Negative").mean()),
        pos_share_posts=("Sentiment", lambda s: (s == "Positive").mean()),
    ).reset_index()

    merged = summaries.merge(agg, left_on="bank", right_on="Bank", how="left").drop(columns="Bank")
    merged = merged.sort_values("content", ascending=False).reset_index(drop=True)
    return merged


def gap_vs_peers(summaries: pd.DataFrame, target_bank: str) -> Dict[str, float]:
    if summaries.empty or target_bank not in summaries["bank"].values:
        return {}
    tgt = summaries.loc[summaries["bank"] == target_bank].iloc[0]
    peers = summaries.loc[summaries["bank"] != target_bank]
    return {
        "neg_share": float(tgt["neg_share"]),
        "neg_share_peer_mean": float(peers["neg_share"].mean()) if not peers.empty else 0.0,
        "pos_share": float(tgt["pos_share"]),
        "pos_share_peer_mean": float(peers["pos_share"].mean()) if not peers.empty else 0.0,
        "engagement_rate": float(tgt["engagement_rate"]),
        "engagement_rate_peer_mean": float(peers["engagement_rate"].mean()) if not peers.empty else 0.0,
        "reach": float(tgt["reach"]),
        "reach_peer_mean": float(peers["reach"].mean()) if not peers.empty else 0.0,
    }
