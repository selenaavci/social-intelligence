"""Multi-bank comparison helpers for the Rakip Analizi tab."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from .pipeline import PipelineResult


MIN_BANKS = 2
MAX_BANKS = 5


@dataclass
class ComparisonResult:
    banks: List[str]
    per_bank: List[PipelineResult]
    kpi: pd.DataFrame
    sentiment_long: pd.DataFrame
    source_long: pd.DataFrame
    daily_long: pd.DataFrame
    pain_union: pd.DataFrame
    opportunity_union: pd.DataFrame
    contexts: List[Dict[str, Any]]


def _validate(banks: List[str]) -> None:
    if len(banks) < MIN_BANKS:
        raise ValueError(f"En az {MIN_BANKS} banka seçin.")
    if len(banks) > MAX_BANKS:
        raise ValueError(f"En fazla {MAX_BANKS} banka seçilebilir.")
    if len(set(banks)) != len(banks):
        raise ValueError("Aynı banka birden fazla kez seçilmiş.")


def build_comparison(per_bank: List[PipelineResult]) -> ComparisonResult:
    banks = [r.bank for r in per_bank]
    _validate(banks)

    kpi_rows = []
    for r in per_bank:
        row = r.summary.as_dict()
        # Override summary metrics with filtered analytics so the table
        # reflects the user's bank-author / ad checkbox choices.
        sd = r.sentiment_dist
        row["content"] = int(r.engagement.get("posts", row.get("content", 0)))
        row["reach"] = int(r.engagement.get("total_reach", row.get("reach", 0)))
        row["avg_interaction"] = float(r.engagement.get("avg_interaction", row.get("avg_interaction", 0.0)))
        if not sd.empty:
            row["pos_share"] = float(sd.get("Positive", 0.0))
            row["neu_share"] = float(sd.get("Neutral", 0.0))
            row["neg_share"] = float(sd.get("Negative", 0.0))
        row["anomaly_days"] = r.context.get("anomaly_days", 0)
        row["pain_topics"] = ", ".join(r.context.get("pain_topics", [])[:3])
        row["opportunity_topics"] = ", ".join(r.context.get("opportunity_topics", [])[:3])
        kpi_rows.append(row)
    kpi = pd.DataFrame(kpi_rows)

    sentiment_rows = []
    for r in per_bank:
        for label, share in r.sentiment_dist.items():
            sentiment_rows.append({
                "bank": r.bank,
                "sentiment": label,
                "share": float(share),
            })
    sentiment_long = pd.DataFrame(sentiment_rows)

    source_rows = []
    for r in per_bank:
        if r.sources.empty:
            continue
        for _, row in r.sources.iterrows():
            source_rows.append({
                "bank": r.bank,
                "source": row["Source"],
                "posts": int(row["posts"]),
                "engagement": float(row["engagement"]) if pd.notna(row["engagement"]) else 0.0,
                "negative_share": float(row["neg"]) if pd.notna(row["neg"]) else 0.0,
            })
    source_long = pd.DataFrame(source_rows)

    daily_frames = []
    for r in per_bank:
        if r.daily.empty:
            continue
        d = r.daily.copy()
        d["bank"] = r.bank
        daily_frames.append(d)
    daily_long = (
        pd.concat(daily_frames, ignore_index=True)
        if daily_frames else pd.DataFrame()
    )

    pain_rows = []
    for r in per_bank:
        if r.pain_points.empty:
            continue
        for _, row in r.pain_points.iterrows():
            pain_rows.append({
                "bank": r.bank,
                "kategori": row["Kategori"],
                "posts": int(row["posts"]),
                "negative_share": float(row["negative_share"]),
                "positive_share": float(row["positive_share"]),
            })
    pain_union = pd.DataFrame(pain_rows)

    opp_rows = []
    for r in per_bank:
        if r.opportunities.empty:
            continue
        for _, row in r.opportunities.iterrows():
            opp_rows.append({
                "bank": r.bank,
                "kategori": row["Kategori"],
                "posts": int(row["posts"]),
                "positive_share": float(row["positive_share"]),
                "negative_share": float(row["negative_share"]),
            })
    opportunity_union = pd.DataFrame(opp_rows)

    contexts = [r.context for r in per_bank]

    return ComparisonResult(
        banks=banks,
        per_bank=per_bank,
        kpi=kpi,
        sentiment_long=sentiment_long,
        source_long=source_long,
        daily_long=daily_long,
        pain_union=pain_union,
        opportunity_union=opportunity_union,
        contexts=contexts,
    )
