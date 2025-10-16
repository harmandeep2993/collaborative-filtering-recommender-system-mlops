# syntax=docker/dockerfile:1
# This Dockerfile builds and runs the FastAPI app inside a lightweight container.
# It installs dependencies, copies project files, and starts the API server.

# 1. Use a lightweight Python 3.11 image as the base
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy environment.yml for dependency tracking (not executed here)
COPY environment.yml /app/

# 4. Upgrade pip and install all required Python libraries
# Using pip instead of conda for faster builds
RUN pip install --upgrade pip \
 && pip install numpy pandas scipy scikit-learn fastapi uvicorn[standard] joblib

# 5. Copy source code, models, and processed data into the container
COPY src /app/src
COPY models /app/models
COPY data/processed /app/data/processed

# 6. Expose port 8000 to allow API access
EXPOSE 8000

# 7. Command to start the FastAPI app using Uvicorn
#    --host 0.0.0.0 : makes the app reachable from outside the container
#    --port 8000    : port number for the API
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------
# HOW THIS WORKS:
# When you build and run this image, Docker starts a Python environment,
# installs dependencies, and runs your FastAPI app (app.py inside src/serving/).
# Uvicorn launches the API server on http://0.0.0.0:8000 inside the container.
# Mapping ports with `docker run -p 8000:8000 <image>` lets you access it from
# your local machine at http://localhost:8000.