#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 32
#SBATCH --gres=gpu
#SBATCH --mem=64G
#SBATCH -t 02:00:00
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
# Load required modules
# =======================
module load cesga/system apptainer/1.2.3


# Lustre-based persistent storage paths
# =======================
LUSTRE_BASE="/mnt/lustre/scratch/nlsas/home/csic/cbb/fpc"
TRANSFORMERS_CACHE="$LUSTRE_BASE/.transformers_cache"
HF_CACHE="$LUSTRE_BASE/.hf_cache"
UDOCKER_REPO="$LUSTRE_BASE/.udocker_repo"
APPTAINER_CACHE="$LUSTRE_BASE/.singularity/cache"
APPTAINER_TMP="$LUSTRE_BASE/.singularity/tmp"
APPTAINER_LOCAL_CACHE="$LUSTRE_BASE/.singularity/local"
PIP_CACHE="$LUSTRE_BASE/.pip_cache"

# =======================
# Export environment variables
# =======================
export TRANSFORMERS_CACHE
export HF_HOME="$HF_CACHE"
export UDOCKER_DIR="$UDOCKER_REPO"
export APPTAINER_CACHEDIR="$APPTAINER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_TMP"
export APPTAINER_LOCALCACHEDIR="$APPTAINER_LOCAL_CACHE"
export PIP_CACHE_DIR="$PIP_CACHE"

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
SHARED_MEM_DIR="/tmp/fantasia"
POSTGRESQL_DATA="$SHARED_MEM_DIR/data"
POSTGRESQL_SOCKET="$SHARED_MEM_DIR/socket"
RABBITMQ_DATA_DIR="$STORE/fantasia_rabbitmq"
CONFIG_FILE="$PROJECT_DIR/fantasia/config.yaml"

# Apptainer container images
PGVECTOR_IMAGE="$PROJECT_DIR/pgvector.sif"
RABBITMQ_IMAGE="$PROJECT_DIR/rabbitmq.sif"
FANTASIA_IMAGE="$PROJECT_DIR/fantasia.sif"

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
    apptainer build "$PGVECTOR_IMAGE" docker://pgvector/pgvector:pg16
fi

if [ ! -f "$RABBITMQ_IMAGE" ]; then
    echo "🔨 Building RabbitMQ container..."
    apptainer build "$RABBITMQ_IMAGE" docker://rabbitmq:management
fi

if [ ! -f "$FANTASIA_IMAGE" ]; then
    echo "🔨 Building FANTASIA container..."
    apptainer build --disable-cache "$FANTASIA_IMAGE" docker://frapercan/fantasia:latest
fi

# =======================
# Launch PostgreSQL (pgvector)
# =======================
echo "🚀 Starting PostgreSQL (pgvector)..."

PGDATA_HOST="$STORE/pgdata_pgvector"
PG_RUN="$STORE/pg_run"

rm -rf "$PGDATA_HOST"
mkdir -p "$PGDATA_HOST" "$PG_RUN"

nohup apptainer run \
  --env POSTGRES_USER=usuario \
  --env POSTGRES_PASSWORD=clave \
  --env POSTGRES_DB="$DB_NAME" \
  --bind "$PGDATA_HOST:/var/lib/postgresql/data" \
  --bind "$PG_RUN:/var/run/postgresql" \
  "$PGVECTOR_IMAGE" > postgres.log 2>&1 &

# =======================
# Launch RabbitMQ
# =======================
echo "🚀 Starting RabbitMQ..."

rm -rf "$RABBITMQ_DATA_DIR"
mkdir -p "$RABBITMQ_DATA_DIR"

nohup apptainer run \
  --bind "$RABBITMQ_DATA_DIR:/var/lib/rabbitmq" \
  "$RABBITMQ_IMAGE" > rabbitmq.log 2>&1 &

# =======================
# Wait for services to become available
# =======================
echo "⏳ Waiting for services to initialize..."
sleep 30

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
    fantasia initialize --base_directory /fantasia

apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
    fantasia run --input "$INPUT" --prefix "$OUTPUT" $EXTRA --base_directory /fantasia

# =======================
# Cleanup services on exit
# =======================
cleanup() {
    echo "🧹 Cleaning up background services..."

    if pgrep -f "$POSTGRESQL_DATA" > /dev/null; then
        echo "🛑 Stopping PostgreSQL..."
        pkill -f "$POSTGRESQL_DATA"
    fi

    if pgrep -f "rabbitmq-server" > /dev/null; then
        echo "🛑 Stopping RabbitMQ..."
        pkill -f "rabbitmq-server"
    fi

    if [[ -d "$SHARED_MEM_DIR" ]]; then
        echo "🗑️ Removing shared memory directory: $SHARED_MEM_DIR"
        rm -rf "$SHARED_MEM_DIR"
    fi
}
