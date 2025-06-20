FROM python:3.12-slim

# Install system dependencies required to build sentencepiece
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy project metadata
COPY pyproject.toml poetry.lock ./

# Install project dependencies (without dev)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-root

# Copy project source code
COPY . .

# Install project in editable mode (without dev)
RUN poetry install --no-dev

# Default command
CMD ["python"]
