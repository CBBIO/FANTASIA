Testing and contributions
=========================

.. code-block:: bash

   poetry install
   poetry run pytest
   poetry run task lint
   poetry run task html_docs

Unit tests must not require the multi-gigabyte reference unless explicitly
marked as integration tests. Changes to CLI/config/output behavior require
tests and matching reference documentation. See the repository
``CONTRIBUTING.md`` for pull-request expectations.
