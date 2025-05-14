#!/bin/bash
#SBATCH --job-name=fantasia
#SBATCH --output=fantasia_%A_%a.log
#SBATCH --error=fantasia_%A_%a.err
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

# Load input parameters

# =======================

PARAM_FILE="$REPO_DIR/fantasia_input_list.txt"
IFS=$'\t' read -r INPUT OUTPUT EXTRA <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
echo "❌ ERROR: Entrada no válida en línea $SLURM_ARRAY_TASK_ID del fichero de parámetros"
exit 1
fi

echo "📥 Input file: $INPUT"
echo "📤 Output name: $OUTPUT"
echo "⚙️  Extra params: $EXTRA"

# =======================

# Clone repo if needed

# =======================

if [ ! -d "$REPO_DIR" ]; then
echo "🚀 Cloning FANTASIA from GitHub..."
git clone https://github.com/CBBIO/FANTASIA.git "$REPO_DIR"
fi

cd "$REPO_DIR" || exit 1

# =======================

# Build containers if missing

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

if [ ! -f "$CONFIG_PATH" ]; then
echo "❌ ERROR: config.yaml not found at $CONFIG_PATH"
exit 1
fi

echo "✅ Configuration file found at $CONFIG_PATH"

singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
fantasia initialize

singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
fantasia run --input "$INPUT" --prefix "$OUTPUT" $EXTRA

# =======================

# Cleanup function

# =======================

cleanup() {
echo "🧹 Cleaning up background services..."
if pgrep -f "$DB_DIR" > /dev/null; then
echo "🛑 Stopping PostgreSQL..."
pkill -f "$DB_DIR"
fi
if pgrep -f "rabbitmq-server" > /dev/null; then
echo "🛑 Stopping RabbitMQ..."
pkill -f "rabbitmq-server"
fi
}

trap cleanup EXIT