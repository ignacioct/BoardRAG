# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED=1

# Set a working directory
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port the Gradio app runs on
EXPOSE 7860

# Set the Gradio server name
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the Gradio app
CMD ["python", "app.py"]
