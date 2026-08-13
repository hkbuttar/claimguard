FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    LOKY_MAX_CPU_COUNT=2 \
    CLAIMGUARD_BUNDLE=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY backend backend
COPY deployment deployment
COPY frequency frequency
COPY frontend frontend
COPY integration integration
COPY severity severity
COPY deployment/bundle/data data
COPY deployment/bundle/reports reports
COPY deployment/bundle/manifest.json manifest.json

RUN useradd --create-home --uid 10001 claimguard \
    && chown -R claimguard:claimguard /app
USER claimguard

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "deployment.serve"]
