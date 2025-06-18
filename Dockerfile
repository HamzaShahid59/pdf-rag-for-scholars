# Use Python 3.10.12 as the base image for building dependencies
FROM python:3.10.12 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create virtual environment
RUN python -m venv /app/.venv

# Install dependencies
COPY requirements.txt .
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Final lightweight image
FROM python:3.10.12-slim

WORKDIR /app

# Copy environment and app files
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Use virtual environment Python
ENV PATH="/app/.venv/bin:$PATH"

# Expose Streamlit default port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
