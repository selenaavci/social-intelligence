"""Data summary (rule-based) and LLM analysis helpers.

Credentials are resolved through ``src.config`` which reads from
``st.secrets`` when running on Streamlit Cloud.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from . import config


SYSTEM_PROMPT = (
    "Sen bir kurumsal iletişim ve sosyal medya analistisin. Sana verilen bankaya "
    "ait sosyal medya metriklerini yorumlar; marka algısı, riskler ve fırsatlar "
    "hakkında profesyonel bir ton ile, Türkçe olarak kısa ve net analiz üretirsin."
)


def data_summary(ctx: Dict[str, Any]) -> str:
    bank = ctx.get("bank", "Banka")
    neg = ctx.get("neg_share", 0.0) * 100
    pos = ctx.get("pos_share", 0.0) * 100
    neu = ctx.get("neu_share", 0.0) * 100
    peer_neg = ctx.get("neg_share_peer_mean", 0.0) * 100
    peer_pos = ctx.get("pos_share_peer_mean", 0.0) * 100
    posts = ctx.get("posts", 0)
    reach = ctx.get("reach", 0)
    pain = ", ".join(ctx.get("pain_topics", [])[:3]) or "belirgin bir konu yok"
    opp = ", ".join(ctx.get("opportunity_topics", [])[:3]) or "belirgin bir konu yok"
    anomalies = ctx.get("anomaly_days", 0)

    parts = [
        f"{bank} son dönemde {posts} sosyal medya gönderisi ile yaklaşık "
        f"{reach:,.0f} erişim elde etmiştir.",
        f"Duygu dağılımı: %{pos:.1f} pozitif, %{neu:.1f} nötr, %{neg:.1f} negatif.",
        f"Sektör ortalamasına göre negatif oran {'yüksek' if neg > peer_neg else 'düşük'} "
        f"(rakip ortalaması %{peer_neg:.1f}); pozitif oran ise "
        f"{'üstünde' if pos > peer_pos else 'altında'} seyrediyor (rakip ortalaması %{peer_pos:.1f}).",
        f"En çok şikayet alan başlıklar: {pain}.",
        f"Öne çıkan olumlu başlıklar: {opp}.",
    ]
    if anomalies:
        parts.append(
            f"Son dönemde {anomalies} günde olağandışı negatif artış tespit edildi; "
            f"kriz izleme mekanizmasının tetiklenmesi önerilir."
        )
    return " ".join(parts)


def _require_llm() -> None:
    if not config.llm_configured():
        raise RuntimeError(
            "LLM yapılandırılmamış. Streamlit Cloud üzerinde Settings -> Secrets "
            "bölümüne LLM_API_KEY, LLM_BASE_URL ve LLM_MODEL değerlerini ekleyin "
            "(lokalde .streamlit/secrets.toml dosyasını kullanın)."
        )


def _chat(user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        timeout=config.LLM_TIMEOUT_S,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content.strip()


def llm_analysis(ctx: Dict[str, Any], user_prompt: str) -> str:
    _require_llm()
    user_payload = json.dumps(ctx, ensure_ascii=False, default=str, indent=2)
    user_message = (
        "Aşağıda seçili bankaya ait sosyal medya metrikleri JSON formatında verilmiştir:\n\n"
        f"{user_payload}\n\n"
        "Kullanıcının özel sorusu / analiz yönü:\n"
        f"{user_prompt.strip()}\n\n"
        "Hem bu metrikleri hem de kullanıcının sorusunu birlikte değerlendirerek "
        "kısa, somut ve uygulanabilir bir yorum üret."
    )
    return _chat(user_message)


def llm_comparison(contexts: List[Dict[str, Any]], user_prompt: str) -> str:
    _require_llm()
    user_payload = json.dumps(contexts, ensure_ascii=False, default=str, indent=2)
    banks = ", ".join(c.get("bank", "?") for c in contexts)
    user_message = (
        f"Aşağıda karşılaştırılacak bankalara ({banks}) ait sosyal medya "
        "metrikleri JSON formatında verilmiştir:\n\n"
        f"{user_payload}\n\n"
        "Kullanıcının özel sorusu / analiz yönü:\n"
        f"{user_prompt.strip()}\n\n"
        "Bu bankaları yan yana karşılaştır; her biri için güçlü ve zayıf yönleri "
        "belirle, sentiment / engagement / reach farklarını yorumla ve somut, "
        "uygulanabilir öneriler sun."
    )
    return _chat(user_message)
