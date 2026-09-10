# Veritas Backend

FastAPI backend untuk Veritas, aplikasi decision intelligence yang menguji asumsi,
unit economics, bukti historis, dan rencana validasi suatu keputusan bisnis.

## Prasyarat

- Python 3.12 atau lebih baru.

## Menjalankan lokal

```powershell
cd AI-Builders-Hackathon-2026-Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Dokumentasi interaktif tersedia di `http://127.0.0.1:8000/docs`; skema mentah di
`/openapi.json`. Frontend Vite lokal (`http://localhost:5173`) sudah menjadi CORS
origin bawaan. Ubah `FRONTEND_ORIGINS` di `.env` untuk origin lain, dengan nilai
dipisahkan koma.

## Endpoint bootstrap

- `GET /health/live` — proses HTTP hidup.
- `GET /health/ready` — bootstrap aplikasi siap.
- `POST /api/v1/analyses` — membuat job analisis berstatus `queued`.
- `GET /api/v1/analyses/{id}` — membaca status job sementara.

Pada Issue 001 job disimpan di memori hanya untuk memvalidasi kontrak. Issue 005
akan menggantinya dengan penyimpanan persisten dan worker background.

## Konfigurasi

Salin `.env.example` menjadi `.env`. Jangan masukkan `.env` atau API key ke Git.
`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, dan `EMBEDDING_MODEL_PATH_OR_ID` belum
digunakan hingga pipeline AI/retrieval diimplementasikan.

## Kualitas kode

```powershell
ruff check .
pytest
```

