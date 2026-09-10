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

## Financial stress test

`app.services.financial.run_financial_stress_test` adalah source of truth untuk unit
economics. Nilai uang dihitung dengan `Decimal(str(input))`, dibulatkan `ROUND_HALF_UP`
ke dua digit hanya saat dipetakan ke API; persentase dan runway dibulatkan ke empat digit.
`runwayMonths` bernilai `null` ketika tidak ada burn dan `breakEvenOrders` bernilai `null`
ketika margin kontribusi per order tidak positif. `FINANCIAL_LOW_RUNWAY_MONTHS` menentukan
ambang warning runway (default 6 bulan).

IDX hanya dibandingkan secara *directional*: contribution-margin ratio dibandingkan dengan
gross margin IDX dan response membawa disclaimer eksplisit. Metrik IDX lain tidak dikirim
karena input saat ini tidak menyediakan accounting debt, assets, atau operating cash flow.

## Konfigurasi

Salin `.env.example` menjadi `.env`. Jangan masukkan `.env` atau API key ke Git.
`EMBEDDING_MODEL_PATH_OR_ID` harus menunjuk ke model SentenceTransformer yang sama dengan
manifest artifact. Untuk deployment offline, letakkan model yang telah diaudit di disk dan
gunakan path tersebut; proses API tidak boleh mengandalkan download model saat startup.

## Menyiapkan artifact retrieval

Raw artifact di `../AI-Builders-Hackathon-2026-Data/Data_Final` belum lengkap untuk runtime:
embedding historical failure dan manifest sengaja tidak dibuat otomatis oleh API. Siapkan
direktori terpisah (yang di-ignore Git) sekali, setelah memverifikasi lisensi dan provenance
dataset serta model:

```powershell
uv run python scripts/prepare_data.py `
  --source-dir ../AI-Builders-Hackathon-2026-Data/Data_Final `
  --output-dir ./prepared-data `
  --model <path-atau-model-id-yang-diaudit> `
  --model-id <model-id> `
  --model-revision <immutable-revision>
```

Set `DATA_DIR=./prepared-data` dan `EMBEDDING_MODEL_PATH_OR_ID` ke model yang sama.
Gunakan `RETRIEVAL_MAX_TOP_K` (1--50, default 10) untuk membatasi jumlah evidence per query.
`prepare_data.py` melakukan scan penuh NaN/Inf, membangun retrieval text dan embedding
historical failure, lalu mencatat checksum, sample checksum, urutan stable ID, dtype, dimensi,
metric, normalisasi, dan model revision dalam `artifact-manifest.json`. Saat startup API hanya
memeriksa manifest dan sample terversi agar tidak menggandakan array atau melakukan full scan.
Readiness tetap `503` sampai artifact dan model telah tervalidasi.

Model sumber notebook harus diaudit sebelum dipakai: pin revision yang immutable dan jangan
mengaktifkan remote custom code. Jalankan benchmark terpisah sebelum memilih FAISS/HNSW; pencatatan
benchmark harus memisahkan waktu load, encode, dan search tanpa menyimpan teks keputusan lengkap.

## Kualitas kode

```powershell
uv run ruff check .
uv run pytest
```

