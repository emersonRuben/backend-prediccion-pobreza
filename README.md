# ENAHO ML API

FastAPI backend to serve predictions from a trained .pkl model.

## Requirements

- Python 3.11+

## Local setup

1. Create virtual environment

```
python -m venv .venv
```

2. Activate

```
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3. Install dependencies

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Add your model file

Place the model at:

```
models/model.pkl
```

5. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

```
docker compose up --build
```

## API

- `GET /health`
- `POST /predict`
- `POST /predict/features`
- `POST /predict/parquet`

### Predict request

```
{
  "records": [[1.2, 3.4, 5.6]]
}
```

### Predict response

```
{
  "predictions": [0.123]
}
```

### Predict with features

You can send a list of dictionaries with column names to `/predict/features`.

```json
{
  "records": [
    {
      "AREA": 1.0,
      "DEPARTAMENTO": 1,
      "ESTRATO": 4
    }
  ]
}
```

### Parquet predict

Send a `.parquet` file in the `file` form field. If you need a fixed column
order, set `FEATURE_COLUMNS` in `.env` (comma-separated).

```

## Configuration

Copy .env.example to .env and adjust values.
```
