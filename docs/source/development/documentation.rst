Build the documentation
=======================

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   sphinx-build -W --keep-going -b html docs/source docs/build/html

Read the Docs uses Python 3.12 and ``docs/requirements.txt``. Keep RST pages in
the toctree, treat warnings as failures, and run
``python scripts/check_documentation.py`` before committing. The internal
claim audit is ``docs/documentation_audit.md``.
