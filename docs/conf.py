# Sphinx configuration for the scijit documentation site.
#
# Build locally:
#     PYTHONPATH=<repo> python -m sphinx -b html -W --keep-going docs _build
#
# Hosted on GitHub Pages, built by .github/workflows/docs.yml in the
# public tree.

import os
import sys

# Make the package importable when Sphinx runs from docs/.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------

project = "SciJIT"
author = "Shmuel Gilbaum"
copyright = "2026, Shmuel Gilbaum"

# The dev tree and the public release carry different versions. The
# projection rewrites the `release` line below; keep it on one line so
# that stays simple.
release = "0.1.4"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",            # Markdown (MyST) source
    "sphinx.ext.autodoc",     # pull docstrings from the importable package
    "sphinx.ext.autosummary", # generate a page per public name
    "numpydoc",               # parse the numpydoc-format docstrings
    "sphinx.ext.intersphinx", # cross-link numpy / numba / scipy
    "sphinx.ext.mathjax",     # render the math in the docstrings
]

# autosummary writes one stub .rst per name into reference/generated/.
autosummary_generate = True

# numpydoc: do not emit a members table of its own (autosummary owns the index),
# and do not fail the build on a docstring that has no matching parameters.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validation_checks = set()

# autodoc: numba dispatchers are neither functions nor classes to inspect, but
# they carry a real __doc__. Document what they expose, keep signatures off the
# ones inspect cannot read.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_typehints = "none"

# autodoc does a REAL import, so the jitclass and dispatcher docstrings are
# introspected. .github/workflows/docs.yml installs gfortran and builds the libraries.

# -- MyST / source --------------------------------------------------------------

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

root_doc = "index"

exclude_patterns = ["_build"]

myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives
    "dollarmath",    # $...$ inline math
    "deflist",
]
# Do not turn bare identifiers with underscores into heading anchors that clash.
myst_heading_anchors = 3

# -- intersphinx ----------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}
# intersphinx needs the network to fetch each objects.inv. RTD has it; a local
# offline build does not, so failures to reach an inventory are non-fatal.
intersphinx_disabled_reftypes = ["*"]

# -- HTML output ----------------------------------------------------------------

html_theme = "furo"
html_title = f"{project} {version}"

# Link the docs back to the source repo: Furo adds a "View this page's source"
# link per page and a GitHub icon in the footer.
html_theme_options = {
    "source_repository": "https://github.com/shmuel-gilbaum/SciJit/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/shmuel-gilbaum/SciJit",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01'
                '-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13'
                '-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 '
                '2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31'
                '-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 '
                '1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 '
                '1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 '
                '3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55'
                '.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

