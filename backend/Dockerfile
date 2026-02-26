FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg and scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY backend/ backend/

# Create necessary directories
RUN mkdir -p data reports/eda reports/segmentation reports/churn reports/forecast

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]