# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fantasia'
copyright = '2024, CBBIO'
author = 'CBBIO'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_static_path = ['_static']
html_title = 'FANTASIA'

#

html_theme = "shibuya"
html_theme_options = {}

autodoc_mock_imports = [
    "yaml", "h5py",
    "Bio", "numpy", "protein_information_system",
    "torch", "pandas", "sklearn", "scipy", "polars", "parasail", "goatools", "ete3",
    "sqlalchemy", "tqdm"
]




master_doc = 'index'

# These DOI targets are valid, but their publishers reject automated HEAD/GET
# requests from Sphinx linkcheck with HTTP 403.
linkcheck_ignore = [
    r"https://doi.org/10\.1093/molbev/msw046",
    r"https://doi.org/10\.1093/nargab/lqae078",
    r"https://doi.org/10\.1002/cpz1\.113",
]
