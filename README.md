# MNI API

FastAPI backend for the Market Narrative Intelligence platform.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy your master database into the data/ folder
mkdir -p data
cp ../mni_project/data/FTSE20_master_database.json data/

# Run the server
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs to see the interactive API documentation.

## Endpoints

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | /health     | Health check and event count         |
| GET    | /sectors    | List all sectors and event counts    |
| GET    | /events     | List all events in the database      |
| POST   | /scenario   | Single scenario assessment           |
| POST   | /compare    | Side-by-side scenario comparison     |
| POST   | /crisis     | Crisis response comparator           |
| POST   | /sector     | Three-view sector analysis           |

## Deployment (Railway)

1. Push this folder to a GitHub repository
2. Connect the repository to Railway
3. Railway will detect the Procfile and deploy automatically
4. Set the DATABASE_PATH environment variable if needed

## Environment Variables

Copy the following from your mni_project .env file:
- TICKER_API_KEY
- NEWSDATA_API_KEY  
- FINNHUB_API_KEY
