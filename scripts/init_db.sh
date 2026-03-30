#!/bin/bash
set -euo pipefail

export PGDATA=/home/user/pgdata
export PGPORT=5433
export PATH=/usr/lib/postgresql/15/bin:$PATH

echo "Attempting PostgreSQL user-space initialization..."

mkdir -p "$PGDATA"

start_postgres=false

if [ -f "$PGDATA/PG_VERSION" ]; then
  start_postgres=true
elif command -v initdb >/dev/null 2>&1; then
  if initdb -D "$PGDATA" --auth=trust --username=user >/tmp/initdb.log 2>&1; then
    start_postgres=true
  else
    echo "initdb failed, switching to SQLite mode"
    export USE_SQLITE=true
  fi
else
  echo "initdb not available, switching to SQLite mode"
  export USE_SQLITE=true
fi

if [ -z "${USE_SQLITE:-}" ] && [ "$start_postgres" = true ]; then
  if command -v pg_ctl >/dev/null 2>&1 && pg_ctl -D "$PGDATA" -l "$PGDATA/logfile" start >/tmp/pg_ctl.log 2>&1; then
    sleep 2
    if pg_ctl -D "$PGDATA" status >/tmp/pg_status.log 2>&1; then
      if command -v createdb >/dev/null 2>&1; then
        createdb -p "$PGPORT" chronomigrate 2>/dev/null || true
      fi
    else
      echo "PostgreSQL failed to reach healthy status, switching to SQLite mode"
      export USE_SQLITE=true
    fi
  else
    echo "PostgreSQL failed to start, switching to SQLite mode"
    export USE_SQLITE=true
  fi
fi

if [ -n "${USE_SQLITE:-}" ]; then
  echo "Running in SQLite fallback mode"
fi

cd /home/user/app
exec python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
