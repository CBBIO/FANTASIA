#!/bin/bash
#SBATCH --job-name=fantasia
#SBATCH --output=fantasia_%j.log
#SBATCH --error=fantasia_%j.err
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =======================
# Load necessary modules
# =======================
module load gcc/13.2.0
module load hdf5/1.14.0
module load singularity/3.11.3


# =======================
# Set directories
# =======================
REPO_DIR="$HOME/FANTASIA"
WORK_DIR="$HOME/FANTASIA"
SHM_DIR="/tmp/fantasia_pgvector"
DB_DIR="$SHM_DIR/data"
DB_SOCKET="$SHM_DIR/socket"
RABBIT_DIR="$HOME/fantasia_rabbitmq"
FANTASIA_RUN_DIR="$HOME/fantasia"
CONFIG_PATH="$REPO_DIR/fantasia/config.yaml"

PGVECTOR_SIF="$WORK_DIR/pgvector_pg18.sif"
RABBITMQ_SIF="$WORK_DIR/rabbitmq.sif"
FANTASIA_SIF="$WORK_DIR/fantasia_v4.1.2.sif"

DB_NAME="BioData"
PG_PORT=5432

# =======================
# Environment check
# =======================
echo "📂 Current working directory:"
pwd
ls -l

# Clone repository if not present
if [ ! -d "$REPO_DIR" ]; then
    echo "🚀 Cloning FANTASIA from GitHub..."
    git clone https://github.com/CBBIO/FANTASIA.git "$REPO_DIR"
fi

cd "$REPO_DIR" || exit 1

# =======================
# Build Singularity containers if missing
# =======================
if [ ! -f "$PGVECTOR_SIF" ]; then
    echo "🔨 Building pgvector.sif..."
    singularity build "$PGVECTOR_SIF" docker://pgvector/pgvector:0.8.1-pg18
fi

if [ ! -f "$RABBITMQ_SIF" ]; then
    echo "🔨 Building rabbitmq.sif..."
    singularity build "$RABBITMQ_SIF" docker://rabbitmq:management
fi

if [ ! -f "$FANTASIA_SIF" ]; then
    echo "🔨 Building fantasia.sif..."
    singularity build --disable-cache "$FANTASIA_SIF" docker://cbbio/fantasia:v4.1.2
fi

# =======================
# Service helpers
# =======================
# Wait until a TCP port on this node accepts connections, or fail after a timeout.
wait_for_port() {  # wait_for_port <name> <port> <timeout_seconds>
    local name=$1 port=$2 timeout=$3 waited=0
    until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
        if (( waited >= timeout )); then
            echo "❌ $name is not accepting connections on port $port after ${timeout}s"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "✅ $name is ready (port $port)"
}

# Send SIGTERM to a background process, wait up to 30 s, then SIGKILL it.
stop_process() {  # stop_process <name> <pid>
    local name=$1 pid=$2
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    echo "🛑 Stopping $name..."
    kill "$pid"
    for _ in $(seq 30); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null
}

# Registered before starting services so they are stopped on any exit,
# including an early exit when fantasia initialize fails.
# =======================
# Cleanup function
# =======================
cleanup() {
    echo "🧹 Cleaning up background services..."
    # pg_ctl restart detaches the server, so stop it through its data directory.
    if [[ -f "$DB_DIR/postmaster.pid" ]]; then
        echo "🛑 Stopping PostgreSQL..."
        singularity exec "$PGVECTOR_SIF" pg_ctl -D "$DB_DIR" stop -m fast -t 60 || true
    fi
    stop_process RabbitMQ "$RABBITMQ_PID"
    if [[ -d "$SHM_DIR" ]]; then
        echo "🗑️ Removing temporary directory: $SHM_DIR"
        rm -rf "$SHM_DIR"
    fi
}

# Ensure cleanup runs on script exit
trap cleanup EXIT

# =======================
# Start PostgreSQL
# =======================
echo "🚀 Starting PostgreSQL..."

rm -rf "$DB_SOCKET" "$DB_DIR"
mkdir -p "$DB_SOCKET" "$DB_DIR"

singularity exec "$PGVECTOR_SIF" initdb -D "$DB_DIR" --encoding=UTF8 --locale=C.UTF-8

cat <<EOF > "$DB_DIR/postgresql.conf"
port = $PG_PORT
unix_socket_directories = '$DB_SOCKET'
EOF

singularity exec "$PGVECTOR_SIF" postgres -D "$DB_DIR" -c config_file="$DB_DIR/postgresql.conf" &
wait_for_port PostgreSQL "$PG_PORT" 300 || exit 1

singularity exec "$PGVECTOR_SIF" psql -h "$DB_SOCKET" -p "$PG_PORT" -d postgres -c "CREATE USER usuario WITH PASSWORD 'clave' SUPERUSER;"
singularity exec "$PGVECTOR_SIF" psql -h "$DB_SOCKET" -p "$PG_PORT" -d postgres -c "CREATE DATABASE \"$DB_NAME\" OWNER usuario;"
singularity exec "$PGVECTOR_SIF" psql -h "$DB_SOCKET" -p "$PG_PORT" -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE \"$DB_NAME\" TO usuario;"

singularity exec "$PGVECTOR_SIF" psql -h "$DB_SOCKET" -p "$PG_PORT" -d "$DB_NAME" -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
singularity exec "$PGVECTOR_SIF" psql -h "$DB_SOCKET" -p "$PG_PORT" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"


singularity exec "$PGVECTOR_SIF" psql -h "$DB_SOCKET" -p "$PG_PORT" -d postgres <<EOF
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET max_worker_processes = 8;
ALTER SYSTEM SET max_parallel_workers = 6;
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET max_connections = 20;
EOF


singularity exec "$PGVECTOR_SIF" pg_ctl -D "$DB_DIR" restart

# =======================
# Start RabbitMQ
# =======================
echo "🚀 Starting RabbitMQ..."

rm -rf "$RABBIT_DIR"
mkdir -p "$RABBIT_DIR"

singularity exec --bind "$RABBIT_DIR:/var/lib/rabbitmq" "$RABBITMQ_SIF" rabbitmq-server &
RABBITMQ_PID=$!
wait_for_port RabbitMQ 5672 300 || exit 1

# =======================
# Run FANTASIA
# =======================
mkdir -p "$FANTASIA_RUN_DIR"

# Check config file
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ ERROR: config.yaml not found at $CONFIG_PATH"
    exit 1
fi

echo "✅ Configuration file found at $CONFIG_PATH"
echo "📂 FANTASIA directory content:"

ls -l "$REPO_DIR/fantasia/"

# Initialize the information system
singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
    fantasia initialize --config "$CONFIG_PATH" --base_directory /fantasia \
    || { echo "❌ fantasia initialize failed"; exit 1; }

# Run analysis
singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
    fantasia run --config "$CONFIG_PATH" --base_directory /fantasia

