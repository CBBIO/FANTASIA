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
    'sphinx_copybutton'
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


html_theme = 'furo'
html_css_files = ["custom.css"]
html_theme_options = {
    "dark_css_variables": {
        "color-sidebar-background": "#DBA7BC",
        "color-toc-background": "#D9A2B8",
        "color-background-primary": "#F8EDF2",
        "color-foreground-primary": "#3c2a30",  # Texto principal en marrón oscuro cálido
        "color-foreground-secondary": "#6e4c59",  # Para secciones menos importantes
        "color-foreground-muted": "#a17888",  # Comentarios o notas suaves
        "color-sidebar-search-background": "#f9eff4",
        "color-link": "#a32c64",  # Rosa fuerte para enlaces
        "color-link--hover": "#6e0047",  # Más oscuro al pasar el ratón

        "color-brand-primary": "#a32c64",  # Mismo rosa para títulos destacados
        "color-brand-content": "#3c2a30",  # Color principal para encabezados

        "color-admonition-title-background": "#f9eff4",  # fondo de las cajas tipo "note"
        "color-admonition-background": "#f8edf2",  # fondo general de advertencias
        # Buscador integrado
        "color-search-border": "#daa5bb",  # borde suave rosa medio 3
        "color-search-text": "#3c2a30",  # texto dentro del input
        "color-search-placeholder": "#a17888",  # texto atenuado (placeholder)

    },
}




master_doc = 'index'
