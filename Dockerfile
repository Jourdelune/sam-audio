FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml .

RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY handler.py .
COPY utils.py .

# Start the handler
CMD [ "python", "-u", "handler.py" ]
