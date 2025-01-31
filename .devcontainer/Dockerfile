FROM mcr.microsoft.com/devcontainers/python:1-3.10

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    python3-dev

RUN mkdir -p /workspace/data/tutorials

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
