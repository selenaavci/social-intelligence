"""Row-level filters for bank-authored posts and advertisements.

Two independent inclusion gates:
- ``is_bank_author``: detects posts written by the analysed bank's own
  account (Odeabank handle, Garanti BBVA handle, ...). Each gate is an
  *inclusion* gate, meaning a post is kept if and only if every gate the
  user enabled allows it.
- ``is_advertisement``: detects posts that disclose themselves as paid /
  promotional content (``#reklam``, ``#sponsorlu``, ``#işbirliği``,
  ``Sponsorlu içerik``, an ``Sponsorluk`` Kategori value, etc.).
"""
from __future__ import annotations

import re
from typing import Iterable, Set

import pandas as pd

# Explicit per-bank handle aliases. Generic slug derivation alone misses
# the short brand names some banks use (e.g. Odeabank -> "odea"), so we
# keep a small override map and fall back to slug derivation.
BANK_AUTHOR_OVERRIDES: dict[str, Set[str]] = {
    "Odeabank": {"odeabank", "odea"},
    "Garanti BBVA": {"garanti", "garantibbva", "garantibank"},
    "Akbank": {"akbank"},
    "Yapı ve Kredi Bankası": {"yapikredi", "yapivekredi", "yapıkredi"},
    "İş Bankası": {"isbankasi", "isbank", "işbankası"},
    "Vakıflar Bankası": {"vakifbank", "vakıfbank"},
    "Vakıf Katılım Bankası": {"vakifkatilim", "vakıfkatılım"},
    "Ziraat Bankası": {"ziraatbank", "ziraatbankasi", "ziraatbankası"},
    "Ziraat Katılım Bankası": {"ziraatkatilim", "ziraatkatılım"},
    "Halk Bankası": {"halkbank", "halkbankasi", "halkbankası"},
    "Kuveyt Türk": {"kuveytturk", "kuveyttürk"},
    "Türkiye Finans Katılım...": {"turkiyefinans", "türkiyefinans"},
    "Albaraka Türk Katılım ...": {"albaraka", "albarakaturk", "albarakatürk"},
    "Denizbank": {"denizbank"},
    "Fibabanka": {"fibabanka"},
    "Şekerbank": {"sekerbank", "şekerbank"},
    "ING Bank": {"ingbank", "ingturkiye", "ing"},
    "QNB Bank": {"qnb", "qnbbank", "qnbfinansbank"},
    "HSBC": {"hsbc", "hsbcturkiye"},
    "ICBC Turkey": {"icbc", "icbcturkey"},
    "Burgan Bank": {"burgan", "burganbank"},
    "Anadolubank": {"anadolubank"},
    "Alternatif Bank": {"alternatifbank"},
    "Aktif Yatırım Bankası": {"aktifbank", "aktifyatirim"},
    "Enpara Bank": {"enpara", "enparabank"},
    "Getir Finans": {"getirfinans"},
    "Midas": {"midas"},
    "Destek Yatırım Bankası": {"destekyatirim", "destek"},
    "Dünya Katılım Bankası": {"dunyakatilim", "dünyakatılım"},
}

_SOCIAL_PLATFORMS = {"X", "Facebook", "Instagram", "TikTok", "LinkedIn", "YouTube"}

_AD_HASHTAGS = (
    "#reklam",
    "#sponsorlu",
    "#sponsored",
    "#işbirliği",
    "#isbirligi",
    "#sponsor",
    "#ad",
    "#paid",
    "#advertising",
    "#advertorial",
)

_AD_PHRASES = (
    "sponsorlu içerik",
    "sponsorlu icerik",
    "advertorial",
    "işbirliği içerir",
    "isbirligi icerir",
)

_AD_DENIALS = (
    "reklam ve işbirliği değildir",
    "reklam ve işbirliği degildir",
    "reklam ve isbirligi degildir",
    "reklam değildir",
    "reklam degildir",
)

_TRAILING_AD_TAG = re.compile(r"\b(reklam|işbirliği|isbirligi)\b\s*[\.\)\]\s]*$", re.IGNORECASE)


def _normalize_token(value: str) -> str:
    """Lowercase and strip punctuation that varies between handles."""
    s = str(value).strip().lower().lstrip("@")
    s = s.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")
    return s


def _slug_aliases(bank: str) -> Set[str]:
    """Derive simple slugs from the bank name as a fallback."""
    s = bank.lower().strip()
    aliases = {s.replace(" ", ""), s.replace(" ", "_")}
    suffixes = (
        "katılım bankası",
        "katilim bankasi",
        "yatırım bankası",
        "yatirim bankasi",
        "bankası",
        "bankasi",
        " bank",
    )
    for suf in suffixes:
        if s.endswith(suf):
            stripped = s[: -len(suf)].strip()
            if stripped:
                aliases.add(stripped.replace(" ", ""))
    return {a for a in aliases if a}


def bank_author_aliases(bank: str) -> Set[str]:
    overrides = BANK_AUTHOR_OVERRIDES.get(bank, set())
    return {_normalize_token(a) for a in (overrides | _slug_aliases(bank)) if a}


def is_bank_author(author, bank: str, aliases: Iterable[str] | None = None) -> bool:
    if pd.isna(author):
        return False
    normalized = _normalize_token(author)
    if not normalized:
        return False
    pool = set(aliases) if aliases is not None else bank_author_aliases(bank)
    for alias in pool:
        if not alias:
            continue
        if normalized == alias:
            return True
        # Match prefix only when followed by a clear boundary so that
        # "odeabank.com.tr" matches "odeabank" but unrelated handles
        # starting with the same letters do not match shorter aliases.
        if normalized.startswith(alias) and len(alias) >= 4:
            tail = normalized[len(alias):]
            if not tail or tail[0].isdigit() or tail.startswith(("com", "tr", "pos", "official", "tr", "bank")):
                return True
    return False


def is_advertisement(row: pd.Series) -> bool:
    text = str(row.get("Text", "") or "").lower()
    kategori = str(row.get("Kategori", "") or "").lower()

    if any(d in text for d in _AD_DENIALS):
        return False
    if any(tag in text for tag in _AD_HASHTAGS):
        return True
    if any(phrase in text for phrase in _AD_PHRASES):
        return True
    if "sponsorluk" in kategori:
        return True
    if _TRAILING_AD_TAG.search(text):
        return True
    return False


def annotate(df: pd.DataFrame, bank: str) -> pd.DataFrame:
    """Add ``is_bank_author`` and ``is_ad`` boolean columns."""
    if df.empty:
        return df.assign(is_bank_author=False, is_ad=False)
    aliases = bank_author_aliases(bank)
    out = df.copy()
    out["is_bank_author"] = out["Author"].apply(
        lambda a: is_bank_author(a, bank, aliases)
    )
    out["is_ad"] = out.apply(is_advertisement, axis=1)
    return out


def apply_filters(
    df: pd.DataFrame,
    bank: str,
    include_bank_authors: bool,
    include_ads: bool,
) -> pd.DataFrame:
    """Drop rows the user opted out of via the dashboard checkboxes."""
    if df.empty:
        return df
    annotated = annotate(df, bank)
    keep = pd.Series(True, index=annotated.index)
    if not include_bank_authors:
        keep &= ~annotated["is_bank_author"]
    if not include_ads:
        keep &= ~annotated["is_ad"]
    return annotated[keep].reset_index(drop=True)


def filter_counts(df: pd.DataFrame, bank: str) -> dict[str, int]:
    """How many posts each gate would remove (for caption display)."""
    if df.empty:
        return {"bank_authors": 0, "ads": 0, "total": 0}
    annotated = annotate(df, bank)
    return {
        "bank_authors": int(annotated["is_bank_author"].sum()),
        "ads": int(annotated["is_ad"].sum()),
        "total": int(len(annotated)),
    }
