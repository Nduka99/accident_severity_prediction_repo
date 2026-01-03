# 1. Base Image
FROM python:3.9-slim

# 2. Set Working Directory
WORKDIR /app

# 2.5 Install System Dependencies (GLib for LightGBM)
# 2.5 Install System Dependencies (GLib for LightGBM) & Upgrade System
RUN apt-get update && apt-get upgrade -y && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 3. Copy Requirements (Backend Only)
COPY requirements-backend.txt .

# 4. Install Dependencies
# Upgrade pip to fix vulnerabilities and install backend deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-backend.txt

# 5. Copy the rest of the application
COPY . .

# 6. Security: Create and Switch to Non-Root User
RUN addgroup --system appgroup && adduser --system --group appuser && \
    chown -R appuser:appgroup /app
USER appuser

# 7. Expose the port (Documenting default, but Render ignores this)
EXPOSE 8000

# 8. Run the application
# Use sh -c to expand environment variables
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
