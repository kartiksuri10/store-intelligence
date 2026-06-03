#!/bin/bash
set -e

echo "=== Store Intelligence Detection Pipeline ==="
echo "Installing dependencies..."
pip install -r pipeline/requirements.txt

echo "Running detection pipeline..."
python pipeline/detect.py \
  --store1-dir "data/Store_1" \
  --store2-dir "data/Store_2" \
  --output "data/events.jsonl" \
  --api-url "http://localhost:8000" \
  --clip-start "2026-03-03T10:00:00Z"

echo "Pipeline complete. Events written to data/events.jsonl"
echo "Checking API health..."
curl -s http://localhost:8000/health | python -m json.tool
