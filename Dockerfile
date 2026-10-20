FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev curl procps && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[rl]" || \
    pip install --no-cache-dir streamlit plotly pandas torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY . .

# Data directory
RUN mkdir -p /data && chmod 777 /data

ENV OSRL_DB=/data/telemetry.db
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Startup: run both collector + dashboard via supervisord
RUN pip install --no-cache-dir supervisor

COPY deploy/supervisord.conf /etc/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
