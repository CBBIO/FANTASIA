Installation and configuration failures
=======================================

``fantasia: command not found``
   Run ``poetry install`` and invoke ``poetry run fantasia``.

Wrong Python version
   Check ``python --version``. This release requires Python 3.12 and rejects
   3.11/3.13 through package metadata.

Config or constants not found
   Run from the repository root or use absolute paths. Check
   ``test -f CONFIG`` and the config's ``constants`` path.

Both partial modes enabled
   Set only one of ``only_lookup`` and ``only_embedding``. The code raises
   ``'only_lookup' and 'only_embedding' cannot both be true.``

No model enabled
   Enable at least one ``embedding.models.<name>.enabled`` entry.
