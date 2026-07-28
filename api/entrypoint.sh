#!/bin/sh
set -e
# A clean checkout has an empty database. Migrate before serving; the seed then
# runs in the app's lifespan and is idempotent.
echo "running migrations..."
alembic upgrade head
exec uvicorn main:app --host 0.0.0.0 --port 8000
