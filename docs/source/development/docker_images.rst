Docker image development
========================

The root Dockerfile builds a Python 3.12 FANTASIA CLI image and installs
PostgreSQL client 16 plus MMseqs2. PostgreSQL and RabbitMQ remain separate
services. Build locally with ``docker build -t fantasia:local .`` and verify
``docker run --rm fantasia:local --help``. Historical publication notes remain
at `Docker Image Build and Publication Guide <../appendix/docker_publish.rst>`_.
