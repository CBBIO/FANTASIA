FROM nvidia/cuda:12.6.1-base-ubuntu24.04

# Update and install required system packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    cd-hit \
    postgresql-client-16 \
    postgresql-contrib \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment for Python
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install Python packages
RUN /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install protein-metamorphisms-is --no-cache-dir


# Add the virtual environment to the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files and set the working directory
COPY . /app
WORKDIR /app

# Default command to keep the container running
ENTRYPOINT ["python3", "-m", "FANTASIA.main"]

