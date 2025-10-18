# -----------------------------------------------
# WindowsContextualSearch - WebAPI + WebUI (Docker)
# -----------------------------------------------

# 1. Base image (Python 3.11-slim)
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the full project
COPY . .

# 6. Expose the default API port
EXPOSE 8000

# 7. Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_ROOT=/app/data

# 8. Command: start backend API and serve web_ui
CMD ["bash", "-c", "uvicorn api.server:app --host 0.0.0.0 --port 8000"]