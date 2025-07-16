# Use a smaller base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the trained model
COPY best_model.pt .

# Copy the application code
COPY stream/ stream/

# Expose the port your application will run on
EXPOSE 8000

# Command to run the application using Gunicorn
CMD ["uvicorn", "stream.server:app", "--workers", "1", "--factory", "--host", "0.0.0.0", "--port", "8000"] 