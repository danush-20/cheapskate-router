FROM python:3.11-slim

WORKDIR /app

# Build deps for llama-cpp-python (compiles a small C++ extension, no GPU needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only build — disables CUDA/Metal so the wheel stays small (~50 MB)
RUN CMAKE_ARGS="-DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF" \
    pip install --no-cache-dir -r requirements.txt

# Download model weights at build time so no network access is needed at grading.
# SmolLM2-1.7B-Instruct Q4_K_M: ~1.1 GB on disk, ~1.3 GB peak RAM.
# Total image budget: python:3.11-slim (~150 MB) + deps (~200 MB) + model (~1.1 GB) ≈ 1.5 GB << 10 GB limit.
RUN mkdir -p /app/models && \
    pip install --no-cache-dir huggingface-hub && \
    python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="bartowski/SmolLM2-1.7B-Instruct-GGUF",
    filename="SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    local_dir="/app/models",
)
EOF

# Rename to the path LOCAL_MODEL_PATH expects
RUN mv /app/models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
       /app/models/smollm2-1.7b-instruct-q4_k_m.gguf

COPY main.py .

ENV INPUT_PATH=/input/tasks.json
ENV OUTPUT_PATH=/output/results.json
ENV LOCAL_MODEL_PATH=/app/models/smollm2-1.7b-instruct-q4_k_m.gguf

CMD ["python", "main.py"]
