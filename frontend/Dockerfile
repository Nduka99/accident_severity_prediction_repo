FROM python:3.9-slim

WORKDIR /app

# 1. Install System Dependencies (Security Updates)
RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy Application Code
COPY . .

# 4. Security: Create and Switch to Non-Root User
RUN addgroup --system appgroup && adduser --system --group appuser && \
    chown -R appuser:appgroup /app

# Fix: Set HOME to a writable directory so apps (Streamlit/Matplotlib) can write configs
ENV HOME=/app

USER appuser

# 5. Expose Streamlit Port
EXPOSE 8501

# 6. Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# 7. Run Application
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
