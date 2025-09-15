# Use a more specific and secure Python image
FROM python:3.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Set the cache directory for Hugging Face transformers to a writable location
ENV TRANSFORMERS_CACHE=/app/cache

# Copy the requirements file and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application files into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 7860

# This is the command to run your app
# It explicitly tells the server to run "app.py" on the correct address and port
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]