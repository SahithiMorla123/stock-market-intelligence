FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces runs containers as a non-root user and needs a
# writable HOME for pip/HF model caches, chroma_db and our own .cache dir.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/chroma_db /app/.cache \
    && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser \
    HF_HOME=/home/appuser/.cache/huggingface

# Hugging Face Spaces (Docker SDK) expects the app on port 7860 by default.
# Render/Railway/etc inject their own $PORT, which takes priority.
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
