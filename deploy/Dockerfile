FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

ENV PATH=/root/.local/bin:$PATH
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
COPY demo/ ./demo/
COPY gcp/ ./gcp/
COPY ingest/ ./ingest/
COPY monitor/ ./monitor/
COPY scripts/ ./scripts/
COPY train/ ./train/
COPY feature_repo/ ./feature_repo/
COPY requirements.txt README.md DESIGN.md ./

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
