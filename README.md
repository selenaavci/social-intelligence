# Social Pulse AI — Streamlit Cloud

Streamlit Cloud üzerinde çalıştırılabilir, bulut-hazır sürüm.

## Deploy adımları

1. Bu `streamlit/` klasörünü kendi GitHub reponuza push edin (kök klasör bu olacak şekilde).
2. <https://share.streamlit.io> adresine gidip **New app** deyin.
3. Repo ve branch'i seçin; **Main file path** olarak `streamlit_app.py` yazın.
4. **Advanced settings -> Secrets** bölümüne aşağıdaki değerleri yapıştırın:

   ```toml
   LLM_API_KEY = "sk-or-v1-..."
   LLM_BASE_URL = "https://openrouter.ai/api/v1"
   LLM_MODEL = "openai/gpt-4o-mini"
   ```

5. Deploy.

## Lokal çalıştırma

```bash
cd streamlit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # değerleri düzenleyin
streamlit run streamlit_app.py
```

## Notlar

- `.streamlit/secrets.toml` repoya commitlenmemeli (varsayılan `.gitignore` bunu engeller).
- Streamlit Cloud'da yazılabilir disk olmadığı için uygulama, yüklenen Excel'i `/tmp/` altında geçici dosyaya yazar ve Excel raporunu bellekte (`BytesIO`) üretir.
- Veri kullanıcı tarafından upload edilir; repoya örnek veri eklenmez.
