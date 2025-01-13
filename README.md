# **FANTASIA**
![FANTASIA Logo](docs/source/_static/FANTASIA.png)

## **Introduction**

FANTASIA (Functional ANnoTAtion based on embedding space SImilArity) is a pipeline for annotating Gene Ontology (GO) terms for protein sequences using advanced protein language models like **ProtT5**, **ProstT5**, and **ESM2**. This system automates complex workflows, from sequence processing to functional annotation, providing a scalable and efficient solution for protein structure and functionality analysis.

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Prerequisites](#prerequisites)
4. [Step 1: Clone the Repository](#step-1-clone-the-repository)
5. [Step 2: Create and Activate a Virtual Environment](#step-2-create-and-activate-a-virtual-environment)
6. [Step 3: Start Services](#step-3-start-services)
7. [Step 4: Configuration](#step-4-configuration)
8. [Step 5: Initialization](#step-5-initialization)
9. [Step 6: Run the Pipeline](#step-6-run-the-pipeline)
10. [Documentation](#documentation)
11. [Citation](#citation)
12. [Contact Information](#contact-information)

---

## **Prerequisites**

1. **Operating System**: Updated Linux (Ubuntu recommended).
2. **Python**: Version 3.10 or higher installed.
3. **Poetry**: Installed for dependency management:
   ```bash
   pip install poetry
   ```
4. **Docker**: Installed and running. If not installed, follow the [Docker installation guide](https://docs.docker.com/get-docker/).
5. **NVIDIA Driver**: Version 550.120 or newer (verify using `nvidia-smi`).
6. **CUDA**: Version 12.4 or newer installed (verify using `nvcc --version`).

---

## **Step 1: Clone the Repository**

```bash
git clone https://github.com/CBBIO/FANTASIA.git
cd FANTASIA
```

---

## **Step 2: Create and Activate a Virtual Environment**

Let `poetry` manage the virtual environment.

```bash
poetry install
poetry shell
```

---

## **Step 3: Start Services**

To ensure the PostgreSQL and RabbitMQ services are running, use the following commands to start the containers:

### **Start PostgreSQL with pgvector**

Run the following command to start a PostgreSQL container with the pgvector extension:

```bash
docker run -d --name pgvectorsql \
    -e POSTGRES_USER=usuario \
    -e POSTGRES_PASSWORD=clave \
    -e POSTGRES_DB=BioData \
    -p 5432:5432 \
    pgvector/pgvector:pg16
```

### **Start RabbitMQ**

Run the following command to start a RabbitMQ container:

```bash
docker run -d --name rabbitmq \
    -p 15672:15672 \
    -p 5672:5672 \
    rabbitmq:management
```

You can access the RabbitMQ management interface at [http://localhost:15672](http://localhost:15672) using the default credentials (`guest`/`guest`).

---

## **Step 4: Configuration**

Before proceeding, create the necessary directories with proper permissions:

```bash
mkdir -p ~/fantasia/dumps ~/fantasia/embeddings ~/fantasia/results ~/fantasia/redundancy
chmod -R 755 ~/fantasia
```

Ensure the following parameters are correctly set in the [config.yaml](fantasia/config.yaml) :


### **System Settings**

```yaml
max_workers: 1
constants: "./fantasia/constants.yaml"  # Auxiliary file for the information system, used to add or remove models in this pipeline.
```

### **PostgreSQL Configuration**

```yaml
DB_USERNAME: usuario
DB_PASSWORD: clave
DB_HOST: pgvectorsql
DB_PORT: 5432
DB_NAME: BioData
```

### **RabbitMQ Configuration**

```yaml
rabbitmq_host: rabbitmq
rabbitmq_user: guest
rabbitmq_password: guest
```

### **Database Dump Source**

```yaml
embeddings_url: "https://zenodo.org/records/14546346/files/embeddings.tar?download=1"
```

### **Paths**

Pay special attention to the paths you configure for FANTASIA:

- **`~/fantasia`**: This is used for input, intermediary, and output files. Ensure that this directory exists and has the correct permissions.
- **`./fantasia`**: Refers to the project root directory where configuration files and scripts reside.

Properly managing these paths ensures smooth execution of the pipeline and prevents errors related to missing files or directories.


```yaml
embeddings_path: ~/fantasia/dumps/
fantasia_output_h5: ~/fantasia/embeddings/
fantasia_output_csv: ~/fantasia/results/
redundancy_file: ~/fantasia/redundancy/output.fasta
```

---

## **Step 5: Initialization**

1. Download embeddings and load the database:

   ```bash
   python fantasia/main.py initialize --config ./fantasia/config.yaml
   ```

2. Verify that the data has been downloaded and loaded into:

   - The folder defined in `embeddings_path`.
   - The configured PostgreSQL database.

---

## **Step 6: Run the Pipeline**


Before running the pipeline, ensure the necessary input file is placed in the correct location. Copy the `zinc_fingers.fasta` file from the `data_sample` directory to the expected input directory:

```bash
mkdir -p ~/fantasia/input
cp ./data_sample/zinc_fingers.fasta ~/fantasia/input/zinc_fingers.fasta
```

Run the pipeline using an input FASTA file and the following command:

```bash
python fantasia/main.py run \
  --fasta ~/fantasia/input/zinc_fingers.fasta \
  --prefix finger_zinc \
  --length_filter 5000 \
  --redundancy_filter 0.65 \
  --sequence_queue_package 200 \
  --esm \
  --prost \
  --prot \
  --distance_threshold 1:1.2,2:0.7,3:0.7 \
  --batch_size 1:50,2:60,3:40
```

### **Explanation of Parameters**

- **`--fasta`**: Specifies the path to the input FASTA file containing protein sequences. In this case: `~/fantasia/input/zinc_fingers.fasta`.
- **`--prefix`**: Sets the prefix for naming the output files. Here, the prefix is `finger_zinc`.
- **`--length_filter`**: Filters out sequences longer than 5000 amino acids.
- **`--redundancy_filter`**: Removes redundant sequences with a similarity threshold of 0.65.
- **`--sequence_queue_package`**: Defines the number of sequences to be processed per queue package (e.g., 200 sequences).
- **`--esm`, `--prost`, `--prot`**: Enables the use of the specified models (ESM, Prost, Prot).
- **`--distance_threshold`**: Sets the maximum allowed distances for similarity matching, specific to each model. Here:
  - Model 1 (ESM): 1.2
  - Model 2 (Prost): 0.7
  - Model 3 (Prot): 0.7
- **`--batch_size`**: Specifies the batch sizes for embedding generation, tailored per model. Here:
  - Model 1 (ESM): 50
  - Model 2 (Prost): 60
  - Model 3 (Prot): 40

### **Output**

Results will be stored in the paths specified under:
- `fantasia_output_h5`: HDF5 embeddings.
- `fantasia_output_csv`: Processed results.



## **Documentation**
(Work In Progress)

For complete details on pipeline configuration, parameters, and deployment, visit the [FANTASIA Documentation](https://protein-metamorphisms-is.readthedocs.io/en/latest/pipelines/fantasia.html).

---

## **Citation**

If you use FANTASIA in your work, please cite the following:

1. Martínez-Redondo, G. I., Barrios, I., Vázquez-Valls, M., Rojas, A. M., & Fernández, R. (2024). Illuminating the functional landscape of the dark proteome across the Animal Tree of Life.  
   https://doi.org/10.1101/2024.02.28.582465.

2. Barrios-Núñez, I., Martínez-Redondo, G. I., Medina-Burgos, P., Cases, I., Fernández, R. & Rojas, A.M. (2024). Decoding proteome functional information in model organisms using protein language models.  
   https://doi.org/10.1101/2024.02.14.580341.

---

## **Contact Information**

- Francisco Miguel Pérez Canales: fmpercan@upo.es  
- Gemma I. Martínez-Redondo: gemma.martinez@ibe.upf-csic.es  
- Ana M. Rojas: a.rojas.m@csic.es  
- Rosa Fernández: rosa.fernandez@ibe.upf-csic.es

