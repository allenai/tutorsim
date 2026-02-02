"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Project information
project = "tutor-bench"
copyright = "2024, Kyle Lo"
author = "Kyle Lo"

# Version info
from tutor_bench.version import __version__  # noqa: E402

release = __version__
version = __version__

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Napoleon configuration (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx configuration (link to other projects' docs)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# MyST configuration (for Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]
myst_heading_anchors = 3

# HTML output configuration
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
}

# Source suffix configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
