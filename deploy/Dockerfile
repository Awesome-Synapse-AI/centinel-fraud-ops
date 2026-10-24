FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.deploy.txt .
RUN pip install --user --no-cache-dir -r requirements.deploy.txt

FROM python:3.11-slim

ENV PATH=/root/.local/bin:$PATH
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
COPY config/ ./config/
COPY demo/ ./demo/
COPY gcp/ ./gcp/
COPY ingest/ ./ingest/
COPY monitor/ ./monitor/
COPY scripts/ ./scripts/
COPY train/ ./train/
COPY feature_repo/ ./feature_repo/
COPY requirements.deploy.txt README.md DESIGN.md ./

EXPOSE 8080
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--bind", "0.0.0.0:8080"]
