#!/bin/bash
set -e

export PGDATA=/home/user/pgdata
export PGPORT=5433
export PATH=/usr/lib/postgresql/15/bin:$PATH

echo "Attempting PostgreSQL user-space initialization..."

if [ ! -f "$PGDATA/PG_VERSION" ]; then
  mkdir -p "$PGDATA"
  if ! initdb -D "$PGDATA" --auth=trust --username=user >/tmp/initdb.log 2>&1; then
    echo "initdb failed, switching to SQLite mode"
    export USE_SQLITE=true
  fi
fi

if [ -z "$USE_SQLITE" ]; then
  if ! pg_ctl -D "$PGDATA" -l "$PGDATA/logfile" start; then
    echo "PostgreSQL failed to start, switching to SQLite mode"
    export USE_SQLITE=true
  else
    createdb -p "$PGPORT" chronomigrate 2>/dev/null || true
  fi
fi

if [ ! -z "$USE_SQLITE" ]; then
  echo "Running in SQLite fallback mode"
fi

cd /home/user/app
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
