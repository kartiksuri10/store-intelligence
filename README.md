# Purplle Tech Challenge - Store Intelligence Pipeline

This repository contains an end-to-end computer vision pipeline and backend API for processing store CCTV footage into real-time retail analytics.

## 1. Setup (5 Commands)

To run the full stack locally:

```bash
git clone https://github.com/purplletechchallenge2026/store-intelligence.git
cd store-intelligence
docker compose up -d
pip install -r pipeline/requirements.txt
python pipeline/detect.py --store1-dir data/Store_1 --store2-dir data/Store_2 --output data/events.jsonl --api-url http://localhost:8000
```

*Note: The detection pipeline processes the clips in the `data/` directory and streams the output directly into the API while saving a local copy to `data/events.jsonl`.*

## 2. Evaluation Deliverables

* **`data/events.jsonl`**: The JSONL event stream perfectly conforms to the provided schema catalogue.
* **`DESIGN.md`**: Contains the AI-Assisted Decisions, Architecture Overview, and Edge Case handling.
* **`CHOICES.md`**: Contains rationales for YOLOv8 (speed), OpenCLIP (zero-shot capability), OSNet (re-id), and FastAPI (async).
* **Idempotency & Edge Cases**: Validated in `app/tests/test_api.py`.
* **Structured Logging**: View API logs via `docker compose logs api`. Logs are emitted as JSON including `trace_id` and `event_count`.
