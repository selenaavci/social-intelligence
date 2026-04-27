"""Social Pulse AI - Streamlit Cloud entry point.

Deploy:
  1. Push this ``streamlit/`` folder as the repository root.
  2. On share.streamlit.io create a new app, set the main file to
     ``streamlit_app.py``.
  3. In Settings -> Secrets add ``LLM_API_KEY``, ``LLM_BASE_URL``,
     ``LLM_MODEL`` (see ``.streamlit/secrets.toml.example``).
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import pipeline
from src.comparison import MAX_BANKS, MIN_BANKS, build_comparison
from src.config import DEFAULT_BANK, llm_configured
from src.data_loader import list_banks, load_all
from src.llm_insights import llm_analysis, llm_comparison
from src.neutral_analysis import (
    category_distribution as neutral_category_distribution,
    find_hidden_positives,
)
from src.pipeline import PipelineResult


st.set_page_config(
    page_title="Social Pulse AI",
    page_icon="🧠",
    layout="wide",
)


# ----- Caching layers ------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_banks(path: str) -> list[str]:
    return list_banks(path)


@st.cache_data(show_spinner=False)
def _cached_load_all(path: str):
    return load_all(path)


@st.cache_resource(show_spinner=False)
def _cached_pipeline(
    path: str,
    bank: str,
    include_bank_authors: bool = True,
    include_ads: bool = True,
) -> PipelineResult:
    combined, summaries = _cached_load_all(path)
    return pipeline.run(
        bank,
        path=path,
        combined=combined,
        summaries=summaries,
        include_bank_authors=include_bank_authors,
        include_ads=include_ads,
    )


def _persist_upload(uploaded) -> str:
    """Write uploaded bytes to a stable /tmp path keyed by content hash."""
    data = uploaded.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:12]
    tmp_path = Path(tempfile.gettempdir()) / f"social_pulse_{digest}.xlsx"
    if not tmp_path.exists():
        tmp_path.write_bytes(data)
    return str(tmp_path)


# ----- Rendering helpers ---------------------------------------------------

def render_detailed(result: PipelineResult) -> None:
    posts = int(result.engagement.get("posts", 0))
    reach = int(result.engagement.get("total_reach", 0))
    sd = result.sentiment_dist
    pos = float(sd.get("Positive", 0.0)) if not sd.empty else 0.0
    neu = float(sd.get("Neutral", 0.0)) if not sd.empty else 0.0
    neg = float(sd.get("Negative", 0.0)) if not sd.empty else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gönderi", f"{posts:,}")
    c2.metric("Erişim", f"{reach:,}")
    c3.metric("Pozitif", f"%{pos * 100:.1f}")
    c4.metric("Nötr", f"%{neu * 100:.1f}")
    c5.metric("Negatif", f"%{neg * 100:.1f}")

    st.divider()

    st.subheader("Veri Özeti")
    st.write(result.data_summary)

    st.subheader("Önerilen Aksiyonlar")
    for act in result.actions:
        st.markdown(f"- {act}")

    st.divider()

    st.subheader("Yapay Zeka ile Birlikte Yorumla")
    st.caption(
        "Veri özetine ek olarak, yapay zekadan metrikler hakkında özel bir yorum "
        "almak ister misiniz? Sormak istediğiniz soruyu veya analiz yönünü aşağıya yazın."
    )
    user_prompt = st.text_area(
        "Yapay zekaya sormak istediğiniz soru",
        placeholder="Örn: Mobil uygulama şikayetlerini bir iletişim planına nasıl dönüştürmeliyim?",
        height=120,
        key="detay_llm_prompt",
    )
    if st.button("Yapay Zeka ile Yorumla", type="primary", key="detay_llm_btn"):
        if not user_prompt.strip():
            st.warning("Lütfen bir soru veya analiz yönü girin.")
        elif not llm_configured():
            st.error(
                "LLM yapılandırılmamış. Streamlit Cloud -> Settings -> Secrets "
                "altına LLM_API_KEY, LLM_BASE_URL, LLM_MODEL ekleyin."
            )
        else:
            with st.spinner("Yapay zeka yanıt oluşturuyor..."):
                try:
                    llm_text = llm_analysis(result.context, user_prompt)
                    st.markdown("#### Yapay Zeka Yorumu")
                    st.write(llm_text)
                except Exception as exc:
                    st.error(f"Yapay zeka çağrısı başarısız oldu: {exc}")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Duygu Dağılımı")
        sd = result.sentiment_dist.rename_axis("sentiment").reset_index(name="share")
        fig = px.pie(sd, names="sentiment", values="share", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Kanal Kırılımı")
        if not result.sources.empty:
            fig = px.bar(
                result.sources, x="Source", y="posts", color="neg",
                color_continuous_scale="Reds", hover_data=["engagement"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Kanal verisi bulunamadı.")

    st.subheader("Günlük Trend ve Anomali Tespiti")
    if not result.anomalies.empty:
        daily = result.anomalies.copy()
        daily["day"] = pd.to_datetime(daily["day"])
        fig = px.line(daily, x="day", y=["posts", "negative"], markers=True)
        anom = daily[daily["is_anomaly"]]
        if not anom.empty:
            fig.add_scatter(
                x=anom["day"], y=anom["negative"], mode="markers",
                marker=dict(size=14, symbol="x", color="red"),
                name="anomali",
            )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Anomali günlerinde en çok konuşulan konular**")
        st.caption(
            "Anomali olarak işaretlenen günlerde gündemi şekillendiren "
            "negatif ve pozitif başlıkları post sayılarıyla görebilirsiniz."
        )
        if not result.anomaly_topics.empty:
            atop = result.anomaly_topics.copy()
            atop["day"] = pd.to_datetime(atop["day"]).dt.strftime("%Y-%m-%d")
            for day in atop["day"].unique():
                st.markdown(f"**{day}**")
                day_df = atop[atop["day"] == day]
                neg = day_df[day_df["sentiment"] == "Negative"][["Kategori", "count"]]
                pos = day_df[day_df["sentiment"] == "Positive"][["Kategori", "count"]]
                cn, cp = st.columns(2)
                with cn:
                    st.markdown("Negatif konular")
                    if not neg.empty:
                        st.dataframe(
                            neg.rename(columns={"Kategori": "Konu", "count": "Post"}),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.caption("Negatif başlık yok.")
                with cp:
                    st.markdown("Pozitif konular")
                    if not pos.empty:
                        st.dataframe(
                            pos.rename(columns={"Kategori": "Konu", "count": "Post"}),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.caption("Pozitif başlık yok.")
        else:
            st.caption("Anomali günlerinde sınıflandırılmış konu bulunamadı.")
    else:
        st.caption("Zaman serisi için yeterli veri yok.")

    st.subheader("Konu ve Ürün İstihbaratı")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**En yüksek negatif oranlı başlıklar**")
        st.dataframe(result.pain_points, use_container_width=True, hide_index=True)
    with t2:
        st.markdown("**En yüksek pozitif oranlı başlıklar**")
        st.dataframe(result.opportunities, use_container_width=True, hide_index=True)

    with st.expander("Tüm konular"):
        st.dataframe(result.topics, use_container_width=True, hide_index=True)

    st.subheader("Kitle Analizi")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Cinsiyet**")
        st.dataframe(result.gender, use_container_width=True, hide_index=True)
    with a2:
        st.markdown("**Influencer vs. Normal Kullanıcı**")
        st.dataframe(result.influencers, use_container_width=True, hide_index=True)

    with st.expander("Etkileşim lideri yazarlar"):
        st.dataframe(result.top_authors, use_container_width=True, hide_index=True)

    st.subheader("Viral Potansiyeli Olan İçerikler")
    st.dataframe(result.viral, use_container_width=True, hide_index=True)

    st.subheader("Rakip Benchmark")
    st.dataframe(result.benchmark, use_container_width=True, hide_index=True)


def render_comparison(comp) -> None:
    st.subheader("Genel KPI Karşılaştırması")
    kpi_display = comp.kpi.copy()
    kpi_display["pos_share"] = (kpi_display["pos_share"] * 100).round(2)
    kpi_display["neu_share"] = (kpi_display["neu_share"] * 100).round(2)
    kpi_display["neg_share"] = (kpi_display["neg_share"] * 100).round(2)
    kpi_display = kpi_display.rename(columns={
        "bank": "Banka", "content": "Gönderi", "reach": "Erişim",
        "pos_share": "Pozitif %", "neu_share": "Nötr %", "neg_share": "Negatif %",
        "engagement_rate": "Engagement Oranı", "avg_interaction": "Ort. Etkileşim",
        "anomaly_days": "Anomali Günü",
        "pain_topics": "Ağrı Noktaları", "opportunity_topics": "Fırsat Başlıkları",
    })
    st.dataframe(kpi_display, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Duygu Dağılımı Karşılaştırması**")
        if not comp.sentiment_long.empty:
            fig = px.bar(
                comp.sentiment_long, x="bank", y="share",
                color="sentiment", barmode="group",
                color_discrete_map={
                    "Positive": "#2ca02c",
                    "Neutral": "#7f7f7f",
                    "Negative": "#d62728",
                },
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Erişim ve Engagement**")
        fig = px.bar(comp.kpi, x="bank", y="reach", title="Toplam Erişim")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(comp.kpi, x="bank", y="engagement_rate", title="Engagement Oranı")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Günlük Gönderi Trendi")
    if not comp.daily_long.empty:
        daily = comp.daily_long.copy()
        daily["day"] = pd.to_datetime(daily["day"])
        fig = px.line(daily, x="day", y="posts", color="bank", markers=True,
                      title="Günlük toplam gönderi")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.line(daily, x="day", y="negative", color="bank", markers=True,
                      title="Günlük negatif gönderi")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Zaman serisi için yeterli veri yok.")

    st.subheader("Kanal Kırılımı")
    if not comp.source_long.empty:
        fig = px.bar(
            comp.source_long, x="source", y="posts", color="bank", barmode="group",
            hover_data=["negative_share", "engagement"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Kanal verisi bulunamadı.")

    st.subheader("Ağrı Noktaları (Negatif Oranı Yüksek Başlıklar)")
    if not comp.pain_union.empty:
        fig = px.bar(
            comp.pain_union, x="kategori", y="negative_share", color="bank",
            barmode="group", hover_data=["posts"],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Tabloyu gör"):
            st.dataframe(comp.pain_union, use_container_width=True, hide_index=True)
    else:
        st.caption("Ağrı noktası verisi bulunamadı.")

    st.subheader("Fırsat Başlıkları (Pozitif Oranı Yüksek)")
    if not comp.opportunity_union.empty:
        fig = px.bar(
            comp.opportunity_union, x="kategori", y="positive_share", color="bank",
            barmode="group", hover_data=["posts"],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Tabloyu gör"):
            st.dataframe(comp.opportunity_union, use_container_width=True, hide_index=True)
    else:
        st.caption("Fırsat verisi bulunamadı.")

    st.divider()

    st.subheader("Yapay Zeka ile Birlikte Yorumla")
    st.caption(
        "Karşılaştırma sonuçlarını yapay zekaya yorumlatmak ister misiniz? "
        "Aşağıya özel sorunuzu veya analiz yönünüzü yazın."
    )
    user_prompt = st.text_area(
        "Yapay zekaya sormak istediğiniz soru",
        placeholder=(
            "Örn: Bu bankaların rekabet konumlandırması nasıl? "
            "Odeabank hangi alanlarda öne geçmeli?"
        ),
        height=120,
        key="rakip_llm_prompt",
    )
    if st.button("Yapay Zeka ile Yorumla", type="primary", key="rakip_llm_btn"):
        if not user_prompt.strip():
            st.warning("Lütfen bir soru veya analiz yönü girin.")
        elif not llm_configured():
            st.error(
                "LLM yapılandırılmamış. Streamlit Cloud -> Settings -> Secrets "
                "altına LLM_API_KEY, LLM_BASE_URL, LLM_MODEL ekleyin."
            )
        else:
            with st.spinner("Yapay zeka yanıt oluşturuyor..."):
                try:
                    llm_text = llm_comparison(comp.contexts, user_prompt)
                    st.markdown("#### Yapay Zeka Yorumu")
                    st.write(llm_text)
                except Exception as exc:
                    st.error(f"Yapay zeka çağrısı başarısız oldu: {exc}")


# ----- Main layout ---------------------------------------------------------

st.sidebar.title("Veri Yükleme")
uploaded = st.sidebar.file_uploader("Excel dosyası yükleyin.", type=["xlsx"])

if not llm_configured():
    st.sidebar.warning(
        "LLM anahtarları eksik. Settings -> Secrets altına LLM_API_KEY, "
        "LLM_BASE_URL ve LLM_MODEL ekleyin."
    )

st.title("Social Intelligence Agent")

if uploaded is None:
    st.info("Analize başlamak için sol menüden bir Excel dosyası yükleyin.")
    st.stop()

data_path = _persist_upload(uploaded)

try:
    banks = _cached_banks(data_path)
except Exception as exc:
    st.error(f"Excel dosyası okunamadı: {exc}")
    st.stop()

if not banks:
    st.error("Dosyada herhangi bir sheet bulunamadı.")
    st.stop()

default_idx = banks.index(DEFAULT_BANK) if DEFAULT_BANK in banks else 0

tab_detay, tab_rakip, tab_nutr = st.tabs(
    ["Detaylı Analiz", "Rakip Analizi", "Nötr Yorumların Analizi"]
)

with tab_detay:
    st.caption("Tek bir bankanın sosyal medya performansını uçtan uca inceleyin.")
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        bank = st.selectbox(
            "Banka seçin",
            options=banks,
            index=default_idx,
            help=f"Önerilen varsayılan: {DEFAULT_BANK}",
            key="detay_bank",
        )
    with col_btn:
        st.write("")
        run_detay = st.button("Analizi Çalıştır", use_container_width=True, key="detay_run")

    bank_label = bank if bank else DEFAULT_BANK
    include_bank_authors = st.checkbox(
        f"{bank_label}'ın yazar olduğu içerikler dahil edilsin",
        value=False,
        key="detay_include_bank_authors",
        help=(
            "Seçili bankanın kendi sosyal medya hesabından yaptığı paylaşımlar "
            "(ör. resmi marka postları) yalnızca bu kutu işaretliyse analize "
            "ve görselleştirmeye katılır."
        ),
    )
    include_ads = st.checkbox(
        "Reklamlar dahil edilsin",
        value=False,
        key="detay_include_ads",
        help=(
            "Reklam olduğu sistem tarafından anlaşılan postlar (#reklam, "
            "#sponsorlu, #işbirliği etiketleri, 'Sponsorlu içerik' ifadesi "
            "veya 'Sponsorluk' kategorisi) yalnızca bu kutu işaretliyse "
            "dahil edilir."
        ),
    )

    if run_detay:
        st.session_state["detay_bank_ran"] = bank
        st.session_state["detay_filters_ran"] = (
            bool(include_bank_authors),
            bool(include_ads),
        )

    active_bank = st.session_state.get("detay_bank_ran")
    if active_bank is None:
        st.info("Bir banka seçin ve Analizi Çalıştır butonuna basın.")
    else:
        active_filters = st.session_state.get("detay_filters_ran", (False, False))
        result = _cached_pipeline(
            data_path,
            active_bank,
            include_bank_authors=active_filters[0],
            include_ads=active_filters[1],
        )
        stats = result.filter_stats
        excluded_bits = []
        if not result.include_bank_authors and stats.get("bank_authors", 0):
            excluded_bits.append(f"{stats['bank_authors']} banka postu")
        if not result.include_ads and stats.get("ads", 0):
            excluded_bits.append(f"{stats['ads']} reklam postu")
        if excluded_bits:
            st.caption("Filtre uygulandı: " + ", ".join(excluded_bits) + " analiz dışında.")
        render_detailed(result)

with tab_rakip:
    st.caption(
        f"En az {MIN_BANKS}, en fazla {MAX_BANKS} banka seçip metriklerini "
        "yan yana karşılaştırın ve sonucu yapay zekaya yorumlatın."
    )

    default_selection = [b for b in (DEFAULT_BANK, "Garanti BBVA", "Akbank") if b in banks][:3]
    if len(default_selection) < MIN_BANKS:
        default_selection = banks[:MIN_BANKS]

    selected = st.multiselect(
        f"Karşılaştırılacak bankalar (en az {MIN_BANKS}, en fazla {MAX_BANKS})",
        options=banks,
        default=default_selection,
        max_selections=MAX_BANKS,
        key="rakip_banks",
    )

    rakip_include_bank_authors = st.checkbox(
        "Bankaların kendi yazar olduğu içerikler dahil edilsin",
        value=False,
        key="rakip_include_bank_authors",
        help=(
            "Karşılaştırılan bankaların kendi sosyal medya hesaplarından "
            "yaptığı paylaşımlar (ör. resmi marka postları) yalnızca bu kutu "
            "işaretliyse analize ve görselleştirmeye katılır."
        ),
    )
    rakip_include_ads = st.checkbox(
        "Reklamlar dahil edilsin",
        value=False,
        key="rakip_include_ads",
        help=(
            "Reklam olduğu sistem tarafından anlaşılan postlar (#reklam, "
            "#sponsorlu, #işbirliği etiketleri, 'Sponsorlu içerik' ifadesi "
            "veya 'Sponsorluk' kategorisi) yalnızca bu kutu işaretliyse "
            "dahil edilir."
        ),
    )

    run_rakip = st.button("Karşılaştırmayı Çalıştır", type="primary", key="rakip_run")

    if run_rakip:
        if len(selected) < MIN_BANKS:
            st.warning(f"Lütfen en az {MIN_BANKS} banka seçin.")
        else:
            st.session_state["rakip_selection_ran"] = tuple(selected)
            st.session_state["rakip_filters_ran"] = (
                bool(rakip_include_bank_authors),
                bool(rakip_include_ads),
            )

    active_selection = st.session_state.get("rakip_selection_ran")
    if active_selection is None:
        st.info("Bankaları seçin ve Karşılaştırmayı Çalıştır butonuna basın.")
    else:
        active_filters = st.session_state.get("rakip_filters_ran", (False, False))
        with st.spinner("Karşılaştırma hazırlanıyor..."):
            per_bank = [
                _cached_pipeline(
                    data_path,
                    b,
                    include_bank_authors=active_filters[0],
                    include_ads=active_filters[1],
                )
                for b in active_selection
            ]
            comp = build_comparison(per_bank)
        excluded_bits = []
        total_bank = sum(r.filter_stats.get("bank_authors", 0) for r in per_bank)
        total_ads = sum(r.filter_stats.get("ads", 0) for r in per_bank)
        if not active_filters[0] and total_bank:
            excluded_bits.append(f"{total_bank} banka postu")
        if not active_filters[1] and total_ads:
            excluded_bits.append(f"{total_ads} reklam postu")
        if excluded_bits:
            st.caption(
                "Filtre uygulandı: " + ", ".join(excluded_bits)
                + " analiz dışında (tüm bankalar toplamı)."
            )
        render_comparison(comp)

with tab_nutr:
    st.caption(
        "Nötr olarak etiketlenmiş yorumların içinde gizli kalan pozitif "
        "geri bildirimleri Türkçe sözlük + emoji puanlamasıyla bulun."
    )

    col_sel_n, col_btn_n = st.columns([3, 1])
    with col_sel_n:
        bank_n = st.selectbox(
            "Banka seçin",
            options=banks,
            index=default_idx,
            key="nutr_bank",
        )
    with col_btn_n:
        st.write("")
        run_nutr = st.button("Analizi Çalıştır", use_container_width=True, key="nutr_run")

    bank_label_n = bank_n if bank_n else DEFAULT_BANK
    nutr_include_bank_authors = st.checkbox(
        f"{bank_label_n}'ın yazar olduğu içerikler dahil edilsin",
        value=False,
        key="nutr_include_bank_authors",
        help=(
            "Seçili bankanın kendi sosyal medya hesabından yaptığı paylaşımlar "
            "yalnızca bu kutu işaretliyse analize katılır."
        ),
    )
    nutr_include_ads = st.checkbox(
        "Reklamlar dahil edilsin",
        value=False,
        key="nutr_include_ads",
        help=(
            "Reklam olduğu sistem tarafından anlaşılan postlar yalnızca bu "
            "kutu işaretliyse dahil edilir."
        ),
    )
    min_score = st.slider(
        "Pozitif eşik (kaç pozitif sinyal yeterli)",
        min_value=1, max_value=5, value=2,
        help="Daha düşük eşik daha fazla aday üretir; daha yüksek eşik daha kesin sonuç verir.",
        key="nutr_min_score",
    )

    if run_nutr:
        st.session_state["nutr_bank_ran"] = bank_n
        st.session_state["nutr_filters_ran"] = (
            bool(nutr_include_bank_authors),
            bool(nutr_include_ads),
            int(min_score),
        )

    active_bank_n = st.session_state.get("nutr_bank_ran")
    if active_bank_n is None:
        st.info("Bir banka seçin ve Analizi Çalıştır butonuna basın.")
    else:
        active_n = st.session_state.get("nutr_filters_ran", (False, False, 2))
        result_n = _cached_pipeline(
            data_path,
            active_bank_n,
            include_bank_authors=active_n[0],
            include_ads=active_n[1],
        )
        df_n = result_n.data
        neutrals_total = int((df_n["Sentiment"].astype(str).str.title() == "Neutral").sum()) if not df_n.empty else 0
        positives = find_hidden_positives(df_n, min_pos_score=active_n[2])
        hidden = int(len(positives))
        share = (hidden / neutrals_total) if neutrals_total > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Nötr Post", f"{neutrals_total:,}")
        c2.metric("Gizli Pozitif", f"{hidden:,}")
        c3.metric("Pay", f"%{share * 100:.1f}")

        if hidden == 0:
            st.info(
                "Bu eşikte gizli pozitif yorum bulunamadı. Eşiği düşürmeyi "
                "veya filtreleri gevşetmeyi deneyebilirsiniz."
            )
        else:
            st.subheader("Kategori Dağılımı")
            cat_df = neutral_category_distribution(positives)
            if not cat_df.empty:
                fig = px.bar(
                    cat_df.head(15),
                    x="posts", y="Kategori", orientation="h",
                    title="Gizli pozitiflerin konu kırılımı",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Kategorilenmiş gizli pozitif yorum yok.")

            st.subheader("Tespit Edilen Pozitif Yorumlar")
            display_cols = [
                c for c in (
                    "Date", "Author", "Source", "Kategori",
                    "pos_score", "neg_score", "matched_pos", "Text", "Link",
                ) if c in positives.columns
            ]
            view = positives[display_cols].copy()
            if "Date" in view.columns:
                view["Date"] = pd.to_datetime(view["Date"]).dt.strftime("%Y-%m-%d")
            view = view.rename(columns={
                "pos_score": "Pozitif Skor",
                "neg_score": "Negatif Skor",
                "matched_pos": "Yakalanan İfadeler",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)
