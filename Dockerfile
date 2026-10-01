# 1. Use the official lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the repository into the container
COPY . .

# 5. Default command to run the complete analysis pipeline
CMD ["python", "analysis.py"]