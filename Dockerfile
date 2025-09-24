FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    gnupg \
    lsb-release \
    ca-certificates \
    mmseqs2

# Install PostgreSQL client 16
RUN apt-get update && \
    apt-get install -y wget gnupg lsb-release && \
    sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/postgres.gpg && \
    apt-get update && \
    apt-get install -y postgresql-client-16 && \
    rm -rf /var/lib/apt/lists/*


# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy project metadata
COPY pyproject.toml ./

# Primero copia
COPY . .

# Luego instala
RUN poetry config virtualenvs.create false \
 && poetry install --no-dev \
 && rm -rf /root/.cache/pypoetry ...


# Default command
ENTRYPOINT ["fantasia"]
