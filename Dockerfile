FROM python:3.11-slim

WORKDIR /app

COPY requirements_backend.txt .

RUN pip install --no-cache-dir -r requirements_backend.txt

COPY backend.py .
COPY kiazi-smart.html .
COPY kiazi.db .
COPY potato_disease_mobilenetv2_best.keras .

EXPOSE 8000


CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000}"]