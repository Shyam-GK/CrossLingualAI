# Use an official Python runtime with Debian (slim version for a smaller image)
FROM python:3.9-slim

# Install FFmpeg and clean up apt cache to keep the image small
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the docker container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (no-cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the necessary temporary directories
RUN mkdir -p uploads outputs

# Define the port as an environment variable (Hugging Face Spaces uses 7860 natively)
ENV PORT=7860
EXPOSE $PORT

# Command to run the application
CMD ["python", "app.py"]
