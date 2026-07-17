# Python base image
FROM python:3.10-slim

# Env vars
ENV PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY Q1/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY Q1/src/ /app/src/
COPY Q1/app.py /app/

# Copy Git and DVC configurations to resolve model tracks inside the container
COPY .git /app/.git
COPY .dvc /app/.dvc

# Expose port for FastAPI
EXPOSE 8000

# Start uvicorn for FastAPI
CMD ["uvicorn", "task_6.main:api", "--host", "0.0.0.0", "--port", "8000"]
