# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.setrecursionlimit(1500)

project = "clusterxplain"
copyright = "2023, Alexander Fottner, Jonathan Pfahler"
author = "Alexander Fottner, Jonathan Pfahler"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinxcontrib.bibtex",
    "sphinx_codeautolink",
    "sphinx.ext.intersphinx",
]

# These projects are also used for the sphinx_codeautolink extension:
intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# autodoc settings
autodoc_member_order = "bysource"


# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = project + " version " + release

# -- Get version information and date from Git ----------------------------

try:
    from subprocess import check_output

    release = check_output(["git", "describe", "--tags", "--always"])
    release = release.decode().strip()
    today = check_output(["git", "show", "-s", "--format=%ad", "--date=short"])
    today = today.decode().strip()
except Exception:
    release = "<unknown>"
    today = "<unknown date>"
