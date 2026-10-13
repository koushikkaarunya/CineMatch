# OCI Ampere A1 is ARM64; the official Python image supports linux/arm64.
FROM --platform=linux/amd64 python:3.14.6-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CINEMATCH_DATA_ROOT=/app/data \
    PORT=8000

WORKDIR /app

COPY requirements-backend.txt ./requirements-backend.txt
RUN python -m pip install --no-cache-dir -r requirements-backend.txt \
    && python -m pip check \
    && useradd --create-home --uid 10001 cinematch

COPY app/ ./app/
COPY scripts/data_paths.py scripts/recommend_collaborative.py scripts/recommend_hybrid.py \
     scripts/recommend_cold_start.py scripts/recommend_hybrid_content.py \
     scripts/recommend_metadata.py scripts/recommend_tfidf.py \
     scripts/recommend_transformer.py ./scripts/
COPY data/enriched/movies.csv ./data/enriched/movies.csv
COPY data/cleaned/movies.csv ./data/cleaned/movies.csv
COPY data/cleaned/ratings.csv ./data/cleaned/ratings.csv
COPY data/models/all_minilm_l6_v2_plots.npz ./data/models/all_minilm_l6_v2_plots.npz

RUN chown -R cinematch:cinematch /app
USER cinematch

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]
