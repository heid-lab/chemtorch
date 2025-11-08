# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

conf_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(conf_dir, "..")
sys.path.insert(0, os.path.join(project_root, "src"))

# Add _extensions to path for custom style
sys.path.insert(0, os.path.join(conf_dir, "_extensions"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ChemTorch"
copyright = f"{datetime.now().year} Esther Heid"
author = "Anton Zamyatin, Jasper De Landsheere, Esther Heid"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ["_templates"]
exclude_patterns = []
extensions = [
    'sphinx.ext.autodoc',       # Core Sphinx autodoc extension
    'sphinx.ext.napoleon',      # For Google-style docstrings
    'sphinx.ext.viewcode',      # Add source code links
    'sphinx.ext.intersphinx',   # Cross-reference other projects
    'sphinx.ext.autosummary',   # Generate summary tables
    'myst_parser',              # For markdown support
    'cli_lexer',                # Custom CLI lexer
]

# Allow embedding program output (we use this to include the `-h` output of scripts in the docs)
extensions.append('sphinxcontrib.programoutput')


# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/heid-lab/chemtorch",
    "use_repository_button": True,
    "logo": {
        "alt_text": "ChemTorch documentation - Home",
        "image_light": "_static/chemtorch_logo_dark_lightbackground.png",
        "image_dark": "_static/chemtorch_logo_dark_lightbackground.png",
    },
}
html_title = "ChemTorch"
html_static_path = ["_static"]
html_css_files = ["custom_syntax.css"]
html_favicon = "_static/chemtorch_favicon.png"
