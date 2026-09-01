FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Data and results are mounted as volumes at runtime — not baked into the image,
# so re-running with new data never requires a rebuild.
ENTRYPOINT ["python"]
CMD ["src/run_pipeline.py"]
