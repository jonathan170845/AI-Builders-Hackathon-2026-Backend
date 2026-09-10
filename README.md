# Veritas Backend

FastAPI backend untuk Veritas, aplikasi decision intelligence yang menguji asumsi,
unit economics, bukti historis, dan rencana validasi suatu keputusan bisnis.

## Prasyarat

- Python 3.12 atau lebih baru.
- `uv` untuk sinkronisasi dependency dari lockfile.

## Menjalankan lokal

```powershell
cd AI-Builders-Hackathon-2026-Backend
uv sync --extra dev --locked
Copy-Item .env.example .env
uv run veritas-api
```

Dokumentasi interaktif tersedia di `http://127.0.0.1:8000/docs`; skema mentah di
`/openapi.json`. Frontend Vite lokal (`http://localhost:5173`) sudah menjadi CORS
origin bawaan. Ubah `FRONTEND_ORIGINS` di `.env` untuk origin lain, dengan nilai
dipisahkan koma. `APP_HOST`, `APP_PORT`, dan `APP_ENV` dibaca oleh entry point;
`APP_ENV=development` mengaktifkan reload.

## Endpoint bootstrap

- `GET /health/live` — proses HTTP hidup.
- `GET /health/ready` — bootstrap aplikasi siap.
- `POST /api/v1/analyses` — membuat job analisis berstatus `queued`.
- `GET /api/v1/analyses/{id}` — membaca status job sementara.

Pada Issue 001 job disimpan di memori hanya untuk memvalidasi kontrak. Issue 005
akan menggantinya dengan penyimpanan persisten dan worker background.

`/health/live` hanya memeriksa process HTTP. `/health/ready` mengembalikan HTTP
`200` saat siap, atau HTTP `503` beserta ringkasan check lokal ketika belum siap.

`financialInputs` opsional. Bila diberikan, seluruh field wajib ada; `monthlyOrders`
adalah integer dan setiap nilai uang dibatasi agar tetap aman untuk kalkulasi berikutnya.
Nilai uang masih unit-agnostic, sehingga frontend tidak boleh mengasumsikan simbol currency.

## Konfigurasi

Salin `.env.example` menjadi `.env`. Jangan masukkan `.env` atau API key ke Git.
`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, dan `EMBEDDING_MODEL_PATH_OR_ID` belum
digunakan hingga pipeline AI/retrieval diimplementasikan.

## Kualitas kode

```powershell
uv run ruff check .
uv run pytest
```

