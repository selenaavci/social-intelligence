"""Audience insight: gender, influencer split, top authors."""
from __future__ import annotations

import pandas as pd

from .config import INFLUENCER_FOLLOWER_THRESHOLD


def gender_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Gender" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("Gender").agg(
        posts=("Text", "size"),
        negative_share=("Sentiment", lambda s: (s == "Negative").mean()),
        positive_share=("Sentiment", lambda s: (s == "Positive").mean()),
        avg_engagement=("Engagement", "mean"),
    ).reset_index().sort_values("posts", ascending=False)
    return g


def influencer_split(df: pd.DataFrame, threshold: int = INFLUENCER_FOLLOWER_THRESHOLD) -> pd.DataFrame:
    if df.empty or "Followers" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["segment"] = (d["Followers"].fillna(0) >= threshold).map(
        {True: "Influencer", False: "Regular"}
    )
    g = d.groupby("segment").agg(
        posts=("Text", "size"),
        total_reach=("Views", "sum"),
        total_interaction=("Interaction", "sum"),
        avg_engagement=("Engagement", "mean"),
        negative_share=("Sentiment", lambda s: (s == "Negative").mean()),
        positive_share=("Sentiment", lambda s: (s == "Positive").mean()),
    ).reset_index()
    return g


def top_authors(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if df.empty or "Author" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("Author").agg(
        posts=("Text", "size"),
        followers=("Followers", "max"),
        total_interaction=("Interaction", "sum"),
        neg_share=("Sentiment", lambda s: (s == "Negative").mean()),
    ).reset_index().sort_values("total_interaction", ascending=False).head(top_k)
    return g.reset_index(drop=True)
