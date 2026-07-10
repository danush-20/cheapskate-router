FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Real paths used by the grading harness. Can still be overridden by the
# harness injecting its own env vars, but these are the documented defaults.
ENV INPUT_PATH=/input/tasks.json
ENV OUTPUT_PATH=/output/results.json

CMD ["python", "main.py"]