# otForge — reproducible CPU-only image.
FROM python:3.13-slim

LABEL org.opencontainers.image.title="otForge"
LABEL org.opencontainers.image.description="Deterministic optimal transport: exact Kantorovich LP, entropic Sinkhorn, Wasserstein barycenters."
LABEL org.opencontainers.image.authors="晨星 (Morning Star)"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

# Determinism + gate + tests in one shot.
RUN python -m pytest tests/ -q
CMD ["python", "-m", "otforge.cli", "run", "--seeds", "0,1,2"]
