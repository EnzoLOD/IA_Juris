FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements files
COPY requirements.txt requirements-dev.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/

# Copy scripts
COPY scripts/ ./scripts/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "src.api.main"]