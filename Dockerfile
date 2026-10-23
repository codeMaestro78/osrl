FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev curl procps && \
    rm -rf /var/lib/apt/lists/*

# Install the package only after its declared source packages exist.
# A failed build must fail visibly, rather than ship a partial RL stack.
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[rl,dashboard]" supervisor

# Data directory
RUN mkdir -p /data && chmod 777 /data

ENV OSRL_DB=/data/telemetry.db
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Startup: run both collector + dashboard via supervisord

COPY deploy/supervisord.conf /etc/supervisord.conf

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
