"""Build an in-memory Excel report for download.

The Streamlit Cloud filesystem is ephemeral, so the report is written to a
``BytesIO`` buffer and returned alongside a suggested filename.
"""
from __future__ import annotations

import io
from typing import Tuple

import pandas as pd

from .pipeline import PipelineResult


def report_filename(result: PipelineResult) -> str:
    safe_bank = result.bank.replace(" ", "_").replace("/", "_")
    return f"{safe_bank}_social_pulse_report.xlsx"


def build_excel_report(result: PipelineResult) -> Tuple[io.BytesIO, str]:
    buffer = io.BytesIO()

    summary_row = pd.DataFrame([result.summary.as_dict()])
    sentiment_df = result.sentiment_dist.rename_axis("sentiment").reset_index(name="share")
    engagement_df = pd.DataFrame([result.engagement])
    summary_text_df = pd.DataFrame({"Veri Özeti": [result.data_summary]})
    actions_df = pd.DataFrame({"Action": result.actions})
    spike_df = pd.DataFrame(result.spike_topics)

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as xl:
        summary_row.to_excel(xl, sheet_name="Summary", index=False)
        sentiment_df.to_excel(xl, sheet_name="Sentiment", index=False)
        engagement_df.to_excel(xl, sheet_name="Engagement", index=False)
        result.sources.to_excel(xl, sheet_name="Sources", index=False)
        result.daily.to_excel(xl, sheet_name="Daily", index=False)
        result.anomalies.to_excel(xl, sheet_name="Anomalies", index=False)
        spike_df.to_excel(xl, sheet_name="Topic Spikes", index=False)
        result.topics.to_excel(xl, sheet_name="Topics", index=False)
        result.pain_points.to_excel(xl, sheet_name="Pain Points", index=False)
        result.opportunities.to_excel(xl, sheet_name="Opportunities", index=False)
        result.gender.to_excel(xl, sheet_name="Gender", index=False)
        result.influencers.to_excel(xl, sheet_name="Influencer Split", index=False)
        result.top_authors.to_excel(xl, sheet_name="Top Authors", index=False)
        result.viral.to_excel(xl, sheet_name="Viral", index=False)
        result.benchmark.to_excel(xl, sheet_name="Benchmark", index=False)
        summary_text_df.to_excel(xl, sheet_name="Veri Ozeti", index=False)
        actions_df.to_excel(xl, sheet_name="Actions", index=False)

    buffer.seek(0)
    return buffer, report_filename(result)
