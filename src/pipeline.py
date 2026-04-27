"""End-to-end pipeline that bundles every analysis artefact for a single bank."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from . import analytics, audience, crisis_detection, llm_insights, recommendations, topic_intelligence
from .data_loader import BankSummary, PathOrBuffer, load_all, load_bank
from .filters import apply_filters, filter_counts


@dataclass
class PipelineResult:
    bank: str
    summary: BankSummary
    data: pd.DataFrame
    sentiment_dist: pd.Series
    engagement: Dict[str, float]
    sources: pd.DataFrame
    daily: pd.DataFrame
    anomalies: pd.DataFrame
    anomaly_topics: pd.DataFrame
    viral: pd.DataFrame
    spike_topics: List[Dict]
    topics: pd.DataFrame
    pain_points: pd.DataFrame
    opportunities: pd.DataFrame
    gender: pd.DataFrame
    influencers: pd.DataFrame
    top_authors: pd.DataFrame
    benchmark: pd.DataFrame
    gap: Dict[str, float]
    data_summary: str
    actions: List[str]
    filter_stats: Dict[str, int] = field(default_factory=dict)
    include_bank_authors: bool = True
    include_ads: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


def run(
    bank: str,
    path: PathOrBuffer,
    combined: Optional[pd.DataFrame] = None,
    summaries: Optional[pd.DataFrame] = None,
    include_bank_authors: bool = True,
    include_ads: bool = True,
) -> PipelineResult:
    df, summary = load_bank(bank, path)
    raw_stats = filter_counts(df, bank)
    df = apply_filters(df, bank, include_bank_authors, include_ads)

    if combined is None or summaries is None:
        combined, summaries = load_all(path)

    sent = analytics.sentiment_distribution(df)
    engagement = analytics.engagement_metrics(df)
    sources = analytics.source_breakdown(df)
    daily = analytics.daily_volume(df)
    anomalies = crisis_detection.detect_anomalies(df)
    anomaly_topics = topic_intelligence.anomaly_day_topics(
        df,
        anomalies.loc[anomalies.get("is_anomaly", False), "day"] if not anomalies.empty else pd.Series(dtype="datetime64[ns]"),
    )
    viral = crisis_detection.viral_content(df)
    spike_topics = crisis_detection.negative_spike_topics(df)
    topics = topic_intelligence.topic_breakdown(df)
    pain = topic_intelligence.top_pain_points(df)
    opps = topic_intelligence.top_opportunities(df)
    gender = audience.gender_breakdown(df)
    inf = audience.influencer_split(df)
    authors = audience.top_authors(df)
    benchmark = analytics.benchmark_banks(combined, summaries)
    gap = analytics.gap_vs_peers(summaries, bank)

    context: Dict[str, Any] = {
        "bank": bank,
        "posts": engagement["posts"],
        "reach": summary.reach,
        "neg_share": summary.neg_share,
        "pos_share": summary.pos_share,
        "neu_share": summary.neu_share,
        "engagement_rate": summary.engagement_rate,
        "avg_interaction": summary.avg_interaction,
        **gap,
        "pain_topics": pain["Kategori"].tolist() if not pain.empty else [],
        "opportunity_topics": opps["Kategori"].tolist() if not opps.empty else [],
        "anomaly_days": int(anomalies["is_anomaly"].sum()) if not anomalies.empty else 0,
        "spike_topics": [s["topic"] for s in spike_topics],
        "top_sources": sources["Source"].head(3).tolist() if not sources.empty else [],
    }

    summary_text = llm_insights.data_summary(context)
    actions = recommendations.generate_recommendations(context)

    return PipelineResult(
        bank=bank,
        summary=summary,
        data=df,
        sentiment_dist=sent,
        engagement=engagement,
        sources=sources,
        daily=daily,
        anomalies=anomalies,
        anomaly_topics=anomaly_topics,
        viral=viral,
        spike_topics=spike_topics,
        topics=topics,
        pain_points=pain,
        opportunities=opps,
        gender=gender,
        influencers=inf,
        top_authors=authors,
        benchmark=benchmark,
        gap=gap,
        data_summary=summary_text,
        actions=actions,
        filter_stats=raw_stats,
        include_bank_authors=include_bank_authors,
        include_ads=include_ads,
        context=context,
    )
