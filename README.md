# Tondo Crime Forecast API (Flask)

This repository packages the Flask API and ARIMA models for Barangay 41 and Barangay 43, so teammates can run the API locally and integrate it with the website.

## Prerequisites
- Python 3.10 or 3.11
- Pip

## Quick Start (Windows PowerShell)

```
cd tondo_crime_api
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Run the API (dev server)
cd app
python run_api.py
```

The server listens on `http://127.0.0.1:5000/`.

Open the test page: `http://127.0.0.1:5000/test_api.html`.

## Endpoints
- `GET /api/locations` → returns `["Barangay 41","Barangay 43"]`
- `GET /api/crime_types` → all crime types with models
- `GET /api/forecast?crime_type=Breaking%20and%20Entering&location=Barangay%2041&months=12` → forecast values
- `GET /api/visualization?crime_type=Breaking%20and%20Entering&location=Barangay%2041&months=12` → chart config + raw data

## Files & Paths
- Models: `tondo_forecasts/models/*.pkl`
- Pre-computed forecasts: `tondo_forecasts/tondo_crime_forecasts.json`
- Historical CSV: `tondo_crime_data_barangay_41_43_2019_2025.csv`
- API code: `app/api.py`, `app/run_api.py`, `app/test_api.html`

The API expects these assets in the repo root relative to `app/`.

## Sharing with Teammates
1. Create a new GitHub repo (e.g., `tondo-crime-api`).
2. Push this folder:
   ```
   cd tondo_crime_api
   git branch -M main
   git remote add origin https://github.com/<your-org>/tondo-crime-api.git
   git push -u origin main
   ```
3. Teammates clone and run the Quick Start above.

## Notes
- Forecast months accepts `12` or `"12 Months"`.
- If you plan to deploy, use `gunicorn` in production and set `--chdir app`.
- Optional Firebase reporting requires `FIREBASE_CREDENTIALS` and `FIREBASE_DB_URL` environment variables.