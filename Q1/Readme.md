# CI Server & Local Deployment Guide

## CI Server Publishing
1. Push commits and create release branch.
2. Train model:
   ```bash
   PYTHONPATH=Q1/src ./.venv/bin/python Q1/app.py t3
   ```
3. Publish new model:
   ```bash
   PYTHONPATH=Q1/src ./.venv/bin/python Q1/app.py t7
   ```
4. Rebuild container:
   ```bash
   docker-compose -f Q1/docker-compose.yml up --build
   ```

---

## Local Development

### 1. Run FastAPI Server
```bash
PYTHONPATH=Q1/src ./.venv/bin/python Q1/app.py t6
```
*   **URL:** http://localhost:8000
*   **Docs:** http://localhost:8000/docs

### 2. Run Tests
```bash
PYTHONPATH=Q1/src ./.venv/bin/pytest Q1/src/task_6/tests/
```

### 3. Docker
*   **Build Image:**
    ```bash
    docker build -t adult-income-api Q1/
    ```
*   **Run Container:**
    ```bash
    docker run -p 8000:8000 adult-income-api
    ```
*   **Compose (Build & Run):**
    ```bash
    docker-compose -f Q1/docker-compose.yml up --build
    ```

---

## Docker Image Sharing

1.  **Save Image :**
    ```bash
    docker save -o adult-income-api.tar adult-income-prediction-api:latest
    ```
2.  **Transfer File:** Move `adult-income-api.tar` to the target machine.
3.  **Load Image **
    ```bash
    docker load -i adult-income-api.tar
    ```
4.  **Run Container :**
    ```bash
    docker run -p 8000:8000 adult-income-prediction-api:latest
    ```