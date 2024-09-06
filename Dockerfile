# Use the official FastAPI Uvicorn image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose port 80
EXPOSE 80

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
