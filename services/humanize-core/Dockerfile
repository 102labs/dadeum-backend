FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY humanize_core ./humanize_core

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "humanize_core.api:app", "--host", "0.0.0.0", "--port", "8000"]

