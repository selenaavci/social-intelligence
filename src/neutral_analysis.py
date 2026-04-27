"""Find positive comments hiding inside the Neutral-labeled bucket.

The upstream sentiment labels in the Somera report are tuned conservatively
- a lot of customer-praise content lands in ``Neutral`` because the model
hesitates between celebratory tone and product mentions. This module runs a
deterministic Turkish lexicon + emoji scorer on top of those Neutral rows
to surface the ones that read as genuinely positive, with the matched
terms surfaced so a human can sanity-check the call.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import pandas as pd


POSITIVE_TERMS: tuple[str, ...] = (
    # Praise
    "harika", "muhteşem", "süper", "mükemmel", "kusursuz", "şahane",
    "favori", "vazgeçilmez", "vazgeçemiyorum", "bayıldım",
    # Satisfaction
    "memnun", "memnuniyet", "memnunum", "beğendim", "beğeniyorum",
    "sevdim", "seviyorum", "tatmin",
    # Gratitude
    "teşekkür", "teşekkürler", "eyvallah", "sağolun", "sağ olun",
    # UX qualities
    "hızlı", "kolay", "pratik", "sorunsuz", "akıcı", "stabil",
    "kullanışlı", "anlaşılır",
    # Outcomes
    "başarılı", "başarı", "kaliteli", "güvenli", "güvenilir",
    "yenilikçi", "fark yaratıyor", "fark yaratan",
    # Recommendation
    "tavsiye ederim", "tavsiye edilir", "öneririm",
    # Value/benefit
    "avantajlı", "avantaj", "fırsat", "kazançlı", "ödüllü",
    "iyi performans", "yüksek performans",
    # General
    "iyi", "çok iyi", "en iyi", "güzel", "çok güzel",
    "kazandım", "ödüllendi", "ödül", "harikaydı",
)

NEGATIVE_TERMS: tuple[str, ...] = (
    "kötü", "berbat", "rezalet", "korkunç", "feci",
    "sorun", "şikayet", "problem", "hata",
    "yetersiz", "eksik", "kötüleşti",
    "yavaş", "açılmıyor", "çalışmıyor", "yüklenmiyor", "donuyor",
    "nefret", "pişman", "pişmanım", "yanıldım",
    "iade", "iptal etmek istiyorum",
    "dolandır", "dolandırıcı", "aldatıldım",
    "zarar", "kayıp", "mağdur",
    "kızgın", "sinir", "bıktım", "bıkkın",
    "kapatıyorum hesab", "müşteri olmaz", "asla",
    "şikayetvar", "şikayet var",
    "değildir", "değil",
)

POSITIVE_EMOJIS: tuple[str, ...] = (
    "👍", "❤️", "❤", "💛", "💙", "💚", "💜", "🧡",
    "😊", "😁", "😄", "🥰", "😍", "🤩",
    "🎉", "🙌", "💪", "⭐", "🌟", "✨", "🚀", "🥇", "👏",
)

NEGATIVE_EMOJIS: tuple[str, ...] = (
    "👎", "😡", "😠", "🤬", "😤", "😞", "😔", "😢", "😭",
    "💔",
)


def _build_pattern(terms: Iterable[str]) -> re.Pattern:
    # Word-boundary aware; honor Turkish letters by treating them as word
    # characters via a manual boundary check around the alternation.
    parts = sorted({re.escape(t) for t in terms}, key=len, reverse=True)
    return re.compile(r"(?<![\wçğıöşüÇĞİÖŞÜ])(" + "|".join(parts) + r")(?![\wçğıöşüÇĞİÖŞÜ])", re.IGNORECASE)


_POS_RE = _build_pattern(POSITIVE_TERMS)
_NEG_RE = _build_pattern(NEGATIVE_TERMS)


def _matches(pattern: re.Pattern, text: str) -> List[str]:
    return [m.group(0).lower() for m in pattern.finditer(text)]


def _emoji_hits(text: str, palette: Iterable[str]) -> List[str]:
    return [e for e in palette if e in text]


def score_text(text: str) -> dict:
    """Return positive/negative score + matched cues for a single post."""
    if not isinstance(text, str) or not text:
        return {"pos_score": 0, "neg_score": 0, "matched_pos": [], "matched_neg": []}
    pos_terms = _matches(_POS_RE, text)
    neg_terms = _matches(_NEG_RE, text)
    pos_emos = _emoji_hits(text, POSITIVE_EMOJIS)
    neg_emos = _emoji_hits(text, NEGATIVE_EMOJIS)
    return {
        "pos_score": len(pos_terms) + len(pos_emos),
        "neg_score": len(neg_terms) + len(neg_emos),
        "matched_pos": pos_terms + pos_emos,
        "matched_neg": neg_terms + neg_emos,
    }


def find_hidden_positives(
    df: pd.DataFrame,
    min_pos_score: int = 2,
    min_lift: float = 2.0,
) -> pd.DataFrame:
    """Return the Neutral rows that read as positive after lexicon scoring.

    A row qualifies when:
      - ``pos_score`` >= ``min_pos_score``, and
      - ``pos_score`` >= ``min_lift`` * ``neg_score`` (strict positivity).
    """
    if df.empty or "Sentiment" not in df.columns or "Text" not in df.columns:
        return pd.DataFrame()

    neutrals = df[df["Sentiment"].astype(str).str.title() == "Neutral"].copy()
    if neutrals.empty:
        return neutrals

    scores = neutrals["Text"].apply(score_text)
    neutrals["pos_score"] = scores.apply(lambda s: s["pos_score"])
    neutrals["neg_score"] = scores.apply(lambda s: s["neg_score"])
    neutrals["matched_pos"] = scores.apply(lambda s: ", ".join(s["matched_pos"][:8]))
    neutrals["matched_neg"] = scores.apply(lambda s: ", ".join(s["matched_neg"][:8]))
    neutrals["positivity"] = neutrals["pos_score"] - neutrals["neg_score"]

    qualifies = (
        (neutrals["pos_score"] >= min_pos_score)
        & (neutrals["pos_score"] >= min_lift * neutrals["neg_score"].clip(lower=0.5))
    )
    out = neutrals[qualifies].sort_values(
        ["positivity", "pos_score"], ascending=[False, False]
    ).reset_index(drop=True)
    return out


def category_distribution(positives: pd.DataFrame) -> pd.DataFrame:
    """Posts-per-Kategori for the detected hidden positives."""
    if positives.empty or "Kategori" not in positives.columns:
        return pd.DataFrame(columns=["Kategori", "posts"])
    d = positives.copy()
    d["Kategori"] = d["Kategori"].astype(str).str.split(r"[,;]")
    d = d.explode("Kategori")
    d["Kategori"] = d["Kategori"].str.strip()
    d = d[d["Kategori"].ne("").fillna(False)]
    d = d[~d["Kategori"].str.lower().isin(["nan", "none", "unknown"])]
    if d.empty:
        return pd.DataFrame(columns=["Kategori", "posts"])
    counts = d.groupby("Kategori").size().reset_index(name="posts")
    return counts.sort_values("posts", ascending=False).reset_index(drop=True)


def summarise(neutrals_total: int, positives_df: pd.DataFrame) -> Tuple[int, int, float]:
    """Returns (neutral_total, hidden_pos_count, share)."""
    hidden = int(len(positives_df))
    share = (hidden / neutrals_total) if neutrals_total > 0 else 0.0
    return neutrals_total, hidden, share
