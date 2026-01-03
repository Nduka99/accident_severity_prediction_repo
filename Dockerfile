# 1. Base Image
FROM python:3.9-slim

# 2. Set Working Directory
WORKDIR /app

# 2.5 Install System Dependencies (GLib for LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 3. Copy Requirements first (for caching)
COPY requirements.txt .

# 4. Install Dependencies
# --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY . .

# 6. Expose the port
EXPOSE 8000

# 7. Run the application
# We use the list format for CMD
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
