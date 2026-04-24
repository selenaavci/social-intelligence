"""Multi-sheet Excel loader for Somera monitoring reports.

Sheet layouts vary across banks (Odeabank carries three preamble rows while
the others carry one), so we locate the header dynamically by scanning for
the row whose first cell equals "Author".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

PathOrBuffer = Union[str, Path, "pd.io.common.ReadCsvBuffer", object]

EXPECTED_COLUMNS = [
    "Author", "Name", "Text", "Link", "Date", "Sentiment",
    "Shares", "Likes", "Comments", "Views", "Followers",
    "Interaction", "Engagement", "Source", "Gender", "Post Type",
    "Attachment", "Official", "Matches", "Kategori", "Tür",
    "Aksiyon", "Other", "ReplyId",
]

NUMERIC_COLUMNS = [
    "Shares", "Likes", "Comments", "Views", "Followers",
    "Interaction", "Engagement",
]


@dataclass
class BankSummary:
    bank: str
    content: int
    avg_interaction: float
    engagement_rate: float
    report_date: str
    reach: int
    neg_share: float
    neu_share: float
    pos_share: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "bank": self.bank,
            "content": self.content,
            "avg_interaction": self.avg_interaction,
            "engagement_rate": self.engagement_rate,
            "report_date": self.report_date,
            "reach": self.reach,
            "neg_share": self.neg_share,
            "neu_share": self.neu_share,
            "pos_share": self.pos_share,
        }


def _coerce_float(x) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")


def _coerce_int(x) -> int:
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return 0


def _find_header_row(raw: pd.DataFrame) -> int:
    for idx in range(min(10, len(raw))):
        first = raw.iat[idx, 0]
        if isinstance(first, str) and first.strip().lower() == "author":
            return idx
    raise ValueError("Could not find 'Author' header row in sheet.")


def _find_summary_row(raw: pd.DataFrame) -> Optional[pd.Series]:
    for idx in range(min(10, len(raw))):
        for col in range(min(10, raw.shape[1])):
            val = raw.iat[idx, col]
            if isinstance(val, str) and val.strip().lower() == "content:":
                return raw.iloc[idx]
    return None


def _extract_summary(bank: str, raw: pd.DataFrame) -> BankSummary:
    row = _find_summary_row(raw)
    if row is None:
        return BankSummary(bank, 0, 0.0, 0.0, "", 0, 0.0, 0.0, 0.0)

    vals = row.tolist()
    mapping: Dict[str, list] = {}
    i = 0
    while i < len(vals):
        cell = vals[i]
        if isinstance(cell, str) and cell.strip().endswith(":"):
            key = cell.strip().rstrip(":").lower()
            payload = []
            j = i + 1
            while j < len(vals):
                nxt = vals[j]
                if isinstance(nxt, str) and nxt.strip().endswith(":"):
                    break
                payload.append(nxt)
                j += 1
            mapping[key] = payload
            i = j
        else:
            i += 1

    sentiment = mapping.get("sentiment", [])
    sentiment = [_coerce_float(v) for v in sentiment[:3]] + [float("nan")] * 3
    neg, neu, pos = sentiment[0], sentiment[1], sentiment[2]

    return BankSummary(
        bank=bank,
        content=_coerce_int((mapping.get("content") or [0])[0]),
        avg_interaction=_coerce_float((mapping.get("avgint") or [0])[0]),
        engagement_rate=_coerce_float((mapping.get("er") or [0])[0]),
        report_date=str((mapping.get("date") or [""])[0]),
        reach=_coerce_int((mapping.get("reach") or [0])[0]),
        neg_share=neg if not np.isnan(neg) else 0.0,
        neu_share=neu if not np.isnan(neu) else 0.0,
        pos_share=pos if not np.isnan(pos) else 0.0,
    )


def _parse_sheet(bank: str, raw: pd.DataFrame) -> pd.DataFrame:
    hdr_idx = _find_header_row(raw)
    df = raw.iloc[hdr_idx + 1:].copy()
    df.columns = raw.iloc[hdr_idx].tolist()
    df = df.loc[:, [c for c in df.columns if isinstance(c, str)]]
    df = df.dropna(how="all").reset_index(drop=True)

    known = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df[known].copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.title()
        df.loc[~df["Sentiment"].isin(["Positive", "Negative", "Neutral"]), "Sentiment"] = "Neutral"
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].fillna("Unknown").astype(str).str.title()
    for c in ("Source", "Post Type", "Kategori", "Official"):
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    df["Bank"] = bank
    return df


def list_banks(path: PathOrBuffer) -> List[str]:
    xl = pd.ExcelFile(path)
    names = list(xl.sheet_names)
    if "Odeabank" in names:
        names.remove("Odeabank")
        names.insert(0, "Odeabank")
    return names


def load_bank(bank: str, path: PathOrBuffer) -> tuple[pd.DataFrame, BankSummary]:
    raw = pd.read_excel(path, sheet_name=bank, header=None)
    df = _parse_sheet(bank, raw)
    summary = _extract_summary(bank, raw)
    return df, summary


def load_all(path: PathOrBuffer) -> tuple[pd.DataFrame, pd.DataFrame]:
    banks = list_banks(path)
    frames: List[pd.DataFrame] = []
    summaries: List[Dict] = []
    for bank in banks:
        try:
            df, summary = load_bank(bank, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Skipping sheet {bank!r}: {exc}")
            continue
        frames.append(df)
        summaries.append(summary.as_dict())
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, pd.DataFrame(summaries)
