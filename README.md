# Hotel Recommender

This repository contains code and API for hotel recommendation system

## Used technologies

- python
- langchain
- docker-compose
- postgresql
- pgvector
- fastapi
- sentence transformers

The recommendation is given with user's query and data in the database (via **similarity search**)

## How to run

### uv:

1. install all dependencies:
```bash
uv sync
```

2. run the database:

```bash
docker-compose up -d
```

3. Add .env file with the connection string
4. store the data from json file to the database

```bash
uv run -m scripts.store_data
```

5. run the backend
```
uvicorn api.main:app --reload
```

### pip:
1. install all dependencies:
```bash
pip install -r requirements.txt
```

2. run the database:

```bash
docker-compose up -d
```

3. Add .env file with the connection string
4. store the data from json file to the database

```bash
python scripts/store_data.py
```

5. run the backend
```
uvicorn api.main:app --reload
```