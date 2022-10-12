import os
import sys

# sys.path.append(os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath("../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "News Recommenders Library"
copyright = "2022, Noel Chia, Ahmed Elzamarany, Nadine Maeser"
author = "Noel Chia, Ahmed Elzamarany, Nadine Maeser"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = "§"
html_theme = "insipid"
html_static_path = ["_static"]

# html_additional_pages = {'index': 'tutorial.html'}

autodoc_default_options = {
    "inherited-members": False,
}
