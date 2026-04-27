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


def anomaly_day_topics(
    df: pd.DataFrame,
    anomaly_days: pd.Series,
    top_k: int = 5,
) -> pd.DataFrame:
    """Most-mentioned positive and negative topics on each anomaly day.

    Returns long-form rows with columns: ``day``, ``sentiment``,
    ``Kategori``, ``count`` ordered by day then count desc within each
    sentiment.
    """
    if df.empty or "Date" not in df.columns or "Kategori" not in df.columns:
        return pd.DataFrame(columns=["day", "sentiment", "Kategori", "count"])

    days = pd.to_datetime(pd.Series(anomaly_days)).dt.floor("D").unique()
    if len(days) == 0:
        return pd.DataFrame(columns=["day", "sentiment", "Kategori", "count"])

    d = df.dropna(subset=["Date"]).copy()
    d["day"] = d["Date"].dt.floor("D")
    d = d[d["day"].isin(days)]
    if d.empty:
        return pd.DataFrame(columns=["day", "sentiment", "Kategori", "count"])

    exploded = d.assign(
        Kategori=d["Kategori"].astype(str).str.split(r"[,;]")
    ).explode("Kategori")
    exploded["Kategori"] = exploded["Kategori"].str.strip()
    exploded = exploded[exploded["Kategori"].ne("").fillna(False)]
    exploded = exploded[~exploded["Kategori"].str.lower().isin(["nan", "none", "unknown"])]
    if exploded.empty:
        return pd.DataFrame(columns=["day", "sentiment", "Kategori", "count"])

    rows = []
    for sentiment in ("Positive", "Negative"):
        sub = exploded[exploded["Sentiment"] == sentiment]
        if sub.empty:
            continue
        counts = (
            sub.groupby(["day", "Kategori"]).size()
            .reset_index(name="count")
            .sort_values(["day", "count"], ascending=[True, False])
        )
        counts["sentiment"] = sentiment
        counts = counts.groupby("day", group_keys=False).head(top_k)
        rows.append(counts[["day", "sentiment", "Kategori", "count"]])

    if not rows:
        return pd.DataFrame(columns=["day", "sentiment", "Kategori", "count"])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["day", "sentiment", "count"], ascending=[True, True, False]
    ).reset_index(drop=True)
