FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR "/app"

COPY Pipfile Pipfile.lock .env* ./

RUN pip install pipenv && pipenv lock && pipenv install --system --deploy

COPY . .

EXPOSE 5000

LABEL authors="Daniel"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]