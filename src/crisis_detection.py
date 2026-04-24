"""Crisis / anomaly detection on daily negative-comment volume."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .analytics import daily_volume
from .config import ANOMALY_CONTAMINATION


def detect_anomalies(df: pd.DataFrame, contamination: float = ANOMALY_CONTAMINATION) -> pd.DataFrame:
    daily = daily_volume(df)
    if daily.empty or len(daily) < 5:
        daily["is_anomaly"] = False
        daily["anomaly_score"] = 0.0
        return daily

    X = daily[["negative", "negative_share", "posts", "interaction"]].to_numpy()
    model = IsolationForest(
        contamination=min(max(contamination, 0.01), 0.4),
        random_state=42,
    )
    labels = model.fit_predict(X)
    daily = daily.copy()
    daily["anomaly_score"] = -model.score_samples(X)
    daily["is_anomaly"] = labels == -1
    return daily


def viral_content(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["viral_score"] = d.get("Views", 0).fillna(0) * np.log1p(d.get("Interaction", 0).fillna(0))
    cols = [c for c in ("Date", "Source", "Sentiment", "Author", "Text", "Views",
                         "Interaction", "Engagement", "Link", "viral_score")
            if c in d.columns]
    return d.sort_values("viral_score", ascending=False)[cols].head(top_k).reset_index(drop=True)


def negative_spike_topics(df: pd.DataFrame, window: int = 1) -> List[Dict]:
    if df.empty or "Kategori" not in df.columns or "Date" not in df.columns:
        return []
    d = df.dropna(subset=["Date"]).copy()
    if d.empty:
        return []
    d["day"] = d["Date"].dt.floor("D")

    latest_day = d["day"].max()
    cutoff = latest_day - pd.Timedelta(days=window - 1)

    exploded = d.assign(
        Kategori=d["Kategori"].astype(str).str.split(r"[,;]")
    ).explode("Kategori")
    exploded["Kategori"] = exploded["Kategori"].str.strip()
    exploded = exploded[exploded["Kategori"].ne("").fillna(False)]

    recent = exploded[exploded["day"] >= cutoff]
    baseline = exploded[exploded["day"] < cutoff]

    results: List[Dict] = []
    for topic, g in recent.groupby("Kategori"):
        if len(g) < 3:
            continue
        recent_neg = (g["Sentiment"] == "Negative").mean()
        base = baseline[baseline["Kategori"] == topic]
        base_neg = (base["Sentiment"] == "Negative").mean() if len(base) else 0.0
        if recent_neg > max(0.15, base_neg * 1.5):
            results.append({
                "topic": topic,
                "recent_posts": int(len(g)),
                "recent_neg_share": float(recent_neg),
                "baseline_neg_share": float(base_neg),
                "lift": float(recent_neg / base_neg) if base_neg > 0 else float("inf"),
            })
    results.sort(key=lambda r: r["recent_neg_share"], reverse=True)
    return results
