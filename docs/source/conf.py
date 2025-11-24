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
copyright = '2024, frapercan'
author = 'frapercan'

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
html_logo = "_static/FANTASIA.png"
html_favicon = "_static/favicon.png"


html_theme = "shibuya"
html_theme_options = {
    "light_logo": "_static/FANTASIA.png",
    "dark_logo": "_static/FANTASIA.png",
}

autodoc_mock_imports = [
    "yaml", "h5py",
    "Bio", "numpy", "protein_information_system",
    "torch", "pandas", "sklearn", "scipy","polars","parasail", "goatools", "ete3"
]




master_doc = 'index'
