"""Action recommendation engine (deterministic, rule-based)."""
from __future__ import annotations

from typing import Any, Dict, List


def generate_recommendations(ctx: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    neg = ctx.get("neg_share", 0.0)
    peer_neg = ctx.get("neg_share_peer_mean", 0.0)
    pain = ctx.get("pain_topics", [])
    opp = ctx.get("opportunity_topics", [])
    anomalies = ctx.get("anomaly_days", 0)
    engagement = ctx.get("engagement_rate", 0.0)
    peer_engagement = ctx.get("engagement_rate_peer_mean", 0.0)

    if neg > max(0.2, peer_neg * 1.2):
        recs.append(
            "Negatif yorum oranı sektör ortalamasının üzerinde: hızlı bir PR "
            "müdahalesi ve müşteri deneyimi iletişim planı hazırlanmalı."
        )
    if pain:
        recs.append(
            f"Şikayet yoğunluğu yüksek başlıklar ({', '.join(pain[:3])}) için "
            "ürün/operasyon ekipleriyle kök neden analizi başlatılmalı."
        )
    if opp:
        recs.append(
            f"Pozitif algı yüksek olan başlıklar ({', '.join(opp[:3])}) "
            "kampanya içeriklerinde öne çıkarılmalı."
        )
    if anomalies:
        recs.append(
            "Anomali tespit edilen günlerde içerik akışı manuel olarak "
            "incelenmeli ve olası kriz uyarıları oluşturularak kapsamları genişletilmeli."
        )
    if engagement < peer_engagement:
        recs.append(
            "Engagement oranı sektör ortalamasının altında: gönderi saati, "
            "format (video/görsel/hikaye) ve CTA denemeleri yapılmalı."
        )
    if not recs:
        recs.append("Mevcut performans stabil; mevcut içerik stratejisini sürdürüp A/B testlerine devam edin.")
    return recs
