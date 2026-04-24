"""Topic and product intelligence from the Kategori column."""
from __future__ import annotations

import pandas as pd

from .config import TOPIC_MIN_POSTS


def _explode_categories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Kategori" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["Kategori"] = d["Kategori"].astype(str).str.split(r"[,;]")
    d = d.explode("Kategori")
    d["Kategori"] = d["Kategori"].str.strip()
    d = d[d["Kategori"].ne("").fillna(False)]
    d = d[~d["Kategori"].str.lower().isin(["nan", "none", "unknown"])]
    return d


def topic_breakdown(df: pd.DataFrame, min_posts: int = TOPIC_MIN_POSTS) -> pd.DataFrame:
    d = _explode_categories(df)
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("Kategori").agg(
        posts=("Text", "size"),
        negative_share=("Sentiment", lambda s: (s == "Negative").mean()),
        positive_share=("Sentiment", lambda s: (s == "Positive").mean()),
        neutral_share=("Sentiment", lambda s: (s == "Neutral").mean()),
        total_interaction=("Interaction", "sum"),
        avg_engagement=("Engagement", "mean"),
    ).reset_index()
    g = g[g["posts"] >= min_posts]
    return g.sort_values("posts", ascending=False).reset_index(drop=True)


def top_pain_points(df: pd.DataFrame, top_k: int = 5, min_posts: int = TOPIC_MIN_POSTS) -> pd.DataFrame:
    g = topic_breakdown(df, min_posts=min_posts)
    if g.empty:
        return g
    return g.sort_values("negative_share", ascending=False).head(top_k).reset_index(drop=True)


def top_opportunities(df: pd.DataFrame, top_k: int = 5, min_posts: int = TOPIC_MIN_POSTS) -> pd.DataFrame:
    g = topic_breakdown(df, min_posts=min_posts)
    if g.empty:
        return g
    return g.sort_values("positive_share", ascending=False).head(top_k).reset_index(drop=True)
