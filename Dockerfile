# QuNtra — Railway deployment image.
#
# One image, two Railway services pointed at it with different Start
# Commands (set per-service in the Railway dashboard):
#   scheduler:    python scripts/scheduler.py --env config/secrets.env --log-file logs/quntra_paper.log
#   telegram-bot: python scripts/run_telegram_bot.py
#
# watchdog.py is intentionally NOT deployed here — its jobs are covered
# differently in the cloud: Railway's own restart-on-crash policy replaces
# its process supervision, the /health endpoint scheduler.py exposes when
# PORT is set replaces its heartbeat-staleness check, and its Docker/
# battery/caffeinate logic is Mac-specific and meaningless on Railway
# (Postgres is a managed Railway plugin here, not a container we control).
FROM python:3.10-slim

# psycopg2-binary needs libpq at runtime; build-essential covers any
# package here that ships without a prebuilt wheel for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-pinned.txt .
RUN pip install --no-cache-dir -r requirements-pinned.txt \
    && pip install --no-cache-dir --no-deps jugaad-data==0.28

COPY . .

RUN mkdir -p logs data/cache data/reports data/backups

# Real value only matters for the healthcheck-serving service (scheduler);
# run_telegram_bot.py never reads PORT, so it's a harmless no-op there.
ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/scheduler.py", "--env", "config/secrets.env", "--log-file", "logs/quntra_paper.log"]
