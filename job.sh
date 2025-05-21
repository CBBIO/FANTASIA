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

PGVECTOR_SIF="$WORK_DIR/pgvector.sif"
RABBITMQ_SIF="$WORK_DIR/rabbitmq.sif"
FANTASIA_SIF="$WORK_DIR/fantasia.sif"

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
    singularity build "$PGVECTOR_SIF" docker://pgvector/pgvector:pg16
fi

if [ ! -f "$RABBITMQ_SIF" ]; then
    echo "🔨 Building rabbitmq.sif..."
    singularity build "$RABBITMQ_SIF" docker://rabbitmq:management
fi

if [ ! -f "$FANTASIA_SIF" ]; then
    echo "🔨 Building fantasia.sif..."
    singularity build --disable-cache "$FANTASIA_SIF" docker://frapercan/fantasia:latest
fi

# =======================
# Start PostgreSQL
# =======================
echo "🚀 Starting PostgreSQL..."

rm -rf "$DB_SOCKET" "$DB_DIR"
mkdir -p "$DB_SOCKET" "$DB_DIR"

singularity exec "$PGVECTOR_SIF" initdb -D "$DB_DIR"

cat <<EOF > "$DB_DIR/postgresql.conf"
port = $PG_PORT
unix_socket_directories = '$DB_SOCKET'
EOF

singularity exec "$PGVECTOR_SIF" postgres -D "$DB_DIR" -c config_file="$DB_DIR/postgresql.conf" &
sleep 5

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
sleep 5

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
    fantasia initialize

# Run analysis
singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
    fantasia run

# =======================
# Cleanup function
# =======================
cleanup() {
    echo "🧹 Cleaning up background services..."

    # Stop PostgreSQL (based on data dir path)
    if pgrep -f "$DB_DIR" > /dev/null; then
        echo "🛑 Stopping PostgreSQL..."
        pkill -f "$DB_DIR"
    fi

    # Stop RabbitMQ if running
    if pgrep -f "rabbitmq-server" > /dev/null; then
        echo "🛑 Stopping RabbitMQ..."
        pkill -f "rabbitmq-server"
    fi

    # Remove temporary directory
    if [[ -d "$SHM_DIR" ]]; then
        echo "🗑️ Removing temporary directory: $SHM_DIR"
        rm -rf "$SHM_DIR"
    fi

}

# Ensure cleanup runs on script exit
trap cleanup EXIT
