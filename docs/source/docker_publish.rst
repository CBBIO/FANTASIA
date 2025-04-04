Docker Image Build and Publication Guide
=========================================

This document describes how to build and publish a Docker image for your project
using a clean, cache-free build process, and push it to Docker Hub with the ``latest`` tag.

Prerequisites
--------------

- Docker must be installed and running.
- A valid ``Dockerfile`` must be present at the root of the project.
- You must have a Docker Hub account (https://hub.docker.com/).
- You must be logged in with ``docker login``.

Clean Build (No Cache)
------------------------

To build the image without using Docker's layer cache:

.. code-block:: bash

    docker build --no-cache -t your-username/your-project:latest .

Replace ``your-username/your-project`` with your actual Docker Hub repository name.

Publishing to Docker Hub
--------------------------

Once the image is built, you can push it to Docker Hub using:

.. code-block:: bash

    docker push your-username/your-project:latest

Verification
-------------

You can verify the image is available online by visiting:

    https://hub.docker.com/r/your-username/your-project

Additional Notes
-----------------

- Ensure your dependencies are up to date and reflected in the appropriate files.
- To free up space from old layers and unused images:

.. code-block:: bash

    docker system prune -f
