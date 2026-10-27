#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 32
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --job-name=fantasia_job
#SBATCH --output=fantasia_%j.out
#SBATCH --error=fantasia_%j.err

################################################################################
# FANTASIA job script for CESGA (SLURM GPU partition)
#
# This script launches the full FANTASIA functional annotation pipeline using:
#
# - PostgreSQL with pgvector (run via Apptainer container)
# - RabbitMQ (run via Apptainer container)
# - FANTASIA (run via Apptainer container with GPU support)
#
# All services are isolated and launched locally within the job node,
# using bind mounts to persistent storage on LUSTRE.
#
# NOTE: This script is intended to be submitted via `sbatch` on CESGA's SLURM system.
################################################################################
# =======================

# Load input parameters

# =======================

PARAM_FILE="./experiments.tsv"
IFS=$'\t' read -r INPUT OUTPUT EXTRA <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
echo "❌ ERROR: Entrada no válida en línea $SLURM_ARRAY_TASK_ID del fichero de parámetros"
exit 1
fi

echo "📥 Input file: $INPUT"
echo "📤 Output name: $OUTPUT"
echo "⚙️  Extra params: $EXTRA"

STORE=$PWD

# =======================
# Define working directories
# =======================
PROJECT_DIR="$STORE/FANTASIA"
EXECUTION_DIR="$STORE/fantasia"
RABBITMQ_DATA_DIR="$STORE/fantasia_rabbitmq_${SLURM_ARRAY_TASK_ID}"
CONFIG_FILE="$PROJECT_DIR/fantasia/config.yaml"

# Apptainer container images
PGVECTOR_IMAGE="$PROJECT_DIR/pgvector_pg18.sif"
RABBITMQ_IMAGE="$PROJECT_DIR/rabbitmq.sif"
FANTASIA_IMAGE="$PROJECT_DIR/fantasia_v4.1.2.sif"

DB_NAME="BioData"
PG_PORT=5432

# =======================
# Environment sanity check
# =======================
echo "📂 Current working directory:"
pwd
ls -l

# Clone FANTASIA repo if missing
if [ ! -d "$PROJECT_DIR" ]; then
    echo "🚀 Cloning FANTASIA from GitHub..."
    git clone https://github.com/CBBIO/FANTASIA.git "$PROJECT_DIR"
fi

cd "$PROJECT_DIR" || exit 1

# =======================
# Build containers if not present
# =======================
if [ ! -f "$PGVECTOR_IMAGE" ]; then
    echo "🔨 Building pgvector container..."
    apptainer build "$PGVECTOR_IMAGE" docker://pgvector/pgvector:0.8.1-pg18
fi

if [ ! -f "$RABBITMQ_IMAGE" ]; then
    echo "🔨 Building RabbitMQ container..."
    apptainer build "$RABBITMQ_IMAGE" docker://rabbitmq:management
fi

if [ ! -f "$FANTASIA_IMAGE" ]; then
    echo "🔨 Building FANTASIA container..."
    apptainer build --disable-cache "$FANTASIA_IMAGE" docker://cbbio/fantasia:v4.1.2
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
# Cleanup services on exit
# =======================
cleanup() {
    echo "🧹 Cleaning up background services..."
    stop_process PostgreSQL "$PG_PID"
    stop_process RabbitMQ "$RABBITMQ_PID"
    # Per-task service data is recreated on every run.
    rm -rf "$PGDATA_HOST" "$PG_RUN" "$RABBITMQ_DATA_DIR"
}

trap cleanup EXIT

# =======================
# Launch PostgreSQL (pgvector)
# =======================
echo "🚀 Starting PostgreSQL (pgvector)..."

# One data directory per array task: tasks must not delete each other's database.
PGDATA_HOST="$STORE/pgdata_pgvector_${SLURM_ARRAY_TASK_ID}"
PG_RUN="$STORE/pg_run_${SLURM_ARRAY_TASK_ID}"

rm -rf "$PGDATA_HOST"
mkdir -p "$PGDATA_HOST" "$PG_RUN"

nohup apptainer run \
  --env POSTGRES_USER=usuario \
  --env POSTGRES_PASSWORD=clave \
  --env POSTGRES_DB="$DB_NAME" \
  --env PGDATA=/var/lib/postgresql/data \
  --env POSTGRES_INITDB_ARGS="--encoding=UTF8 --locale=C.UTF-8" \
  --bind "$PGDATA_HOST:/var/lib/postgresql/data" \
  --bind "$PG_RUN:/var/run/postgresql" \
  "$PGVECTOR_IMAGE" > postgres.log 2>&1 &
PG_PID=$!

# =======================
# Launch RabbitMQ
# =======================
echo "🚀 Starting RabbitMQ..."

rm -rf "$RABBITMQ_DATA_DIR"
mkdir -p "$RABBITMQ_DATA_DIR"

nohup apptainer run \
  --bind "$RABBITMQ_DATA_DIR:/var/lib/rabbitmq" \
  "$RABBITMQ_IMAGE" > rabbitmq.log 2>&1 &
RABBITMQ_PID=$!

# =======================
# Wait for services to become available
# =======================
echo "⏳ Waiting for services to initialize..."
wait_for_port PostgreSQL "$PG_PORT" 600 || exit 1
wait_for_port RabbitMQ 5672 300 || exit 1

# =======================
# Run FANTASIA pipeline
# =======================
mkdir -p "$EXECUTION_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

echo "✅ Found configuration file: $CONFIG_FILE"
echo "📂 Listing FANTASIA contents:"
ls -l "$PROJECT_DIR/fantasia/"

apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
    fantasia initialize --config "$CONFIG_FILE" --base_directory /fantasia \
    || { echo "❌ fantasia initialize failed"; exit 1; }

apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
    fantasia run --config "$CONFIG_FILE" --input "$INPUT" --prefix "$OUTPUT" $EXTRA --base_directory /fantasia

