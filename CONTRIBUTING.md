# Contributing to FANTASIA

Use Python 3.12 and install with `poetry install`. Before a pull request, run:

```bash
poetry run pytest
poetry run task lint
poetry run task html_docs
python scripts/check_documentation.py
```

Keep scientific behavior and documentation changes explicit and separate when
possible. Add tests for CLI, configuration, parser, schema, or output changes.
Update the canonical reference page rather than duplicating long explanations.
Document new models in `fantasia/constants.yaml`, packaged configs, the model
reference, resource notes, and tests. Document every new configuration key with
its type, default, stage, interactions, and example. Do not commit reference
databases, model caches, generated experiments, secrets, or personal paths.

Pull requests should state motivation, implementation, compatibility impact,
validation commands actually run, documentation changes, and any large-data or
GPU checks that could not be performed. Add a concise entry to `CHANGELOG.md`
for user-visible changes. Maintainers control release versions and tags.
