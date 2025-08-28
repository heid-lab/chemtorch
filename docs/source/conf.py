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


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ChemTorch"
copyright = f"{datetime.now().year} Esther Heid"
author = "Jasper De Landsheere, Anton Zamyatin, Esther Heid"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/heid-lab/chemtorch",
    "use_repository_button": True,
    "logo": {
        "alt_text": "ChemTorch documentation - Home",
        "image_light": "_static/chemtorch_logo_light.svg",
        "image_dark": "_static/chemtorch_logo_dark.svg",
    },
}
html_title = "ChemTorch"
html_static_path = ["_static"]
html_favicon = "_static/chemtorch_favicon.png"
