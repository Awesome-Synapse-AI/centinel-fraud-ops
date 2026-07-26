PYTHON ?= python

.PHONY: ingest train score demo test

ingest:
	$(PYTHON) ingest/build_features.py --output-dir data/staged/v1

train:
	$(PYTHON) train/train.py --features data/staged/v1/features.parquet --output outputs/model.joblib

score:
	uvicorn app.main:app --host 0.0.0.0 --port 8080

demo:
	streamlit run demo/streamlit_app.py

test:
	$(PYTHON) -m pytest tests
