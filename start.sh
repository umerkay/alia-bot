#!/bin/bash
set -e

# Run processing step
echo "Processing documents..."
python -m app.services.process_docs shared_docs

# Start FastAPI app
echo "Starting FastAPI..."
exec uvicorn app.main:app --host 0.0.0.0 --port 7860