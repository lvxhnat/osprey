#!/bin/bash

python manage.py makemigrations

python manage.py migrate

python manage.py collectstatic --noinput

gunicorn osprey_admin.asgi:application -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000