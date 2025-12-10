# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sphinx configuration for Aerial Framework documentation.

This configuration file sets up:
- nvidia-sphinx-theme for consistent NVIDIA documentation styling
- Breathe integration for C++ API documentation via Doxygen
- Mermaid support for diagram rendering
- Python autodoc for automated API documentation
- Napoleon for Google-style docstring parsing
"""

# Sphinx configuration with nvidia-sphinx-theme and Breathe integration

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Python source for autodoc
project_root = Path(__file__).resolve().parent.parent
source_root = project_root / "ran" / "py" / "src"

if not source_root.exists():
    logger.error(f"Python source directory does not exist: {source_root}")
    raise FileNotFoundError(f"Python source directory not found: {source_root}")

sys.path.insert(0, str(source_root))

# Read version from VERSION file
version_file = project_root / "VERSION"
with version_file.open() as f:
    version = release = f.read().strip()

# Project information
project = "Aerial Framework"
project_copyright = "2025, NVIDIA CORPORATION"
author = "NVIDIA Aerial"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",  # Added missing extension
    "breathe",
    "sphinxcontrib.mermaid",  # Mermaid diagram support
    "myst_parser",  # Markdown support
    "nbsphinx",  # Jupyter notebook support
]

# Mermaid configuration
mermaid_output_format = "raw"  # Use 'raw' for HTML, 'svg'/'png' for static
mermaid_version = "11.2.0"  # Latest stable version
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'default'});"

# Nbsphinx configuration
nbsphinx_execute = "never"  # Don't re-execute (use pre-executed notebooks)
nbsphinx_allow_errors = False  # Fail build if notebook has errors

# Templates and static files
templates_path = ["_templates"]
html_static_path = ["_static", "figures"]

# Custom CSS files
html_css_files = [
    "custom.css",
]

exclude_patterns = [
    "_build",
    "out/**",
    "**/.doctrees/**",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    ".pytest_cache",
    "src",
    "tests",
    "dist",
    "*.egg-info",
    "uv.lock",
    "internal/**",
]

# HTML theme configuration
html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    # Keep only supported options for nvidia-sphinx-theme
    "collapse_navigation": False,
    "navigation_depth": 4,
}

# Additional HTML options
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True

# Breathe configuration for C++ documentation
breathe_projects = {"aerial-framework": str(Path(__file__).parent / "doxygen" / "xml")}
breathe_default_project = "aerial-framework"
breathe_default_members = ("members", "undoc-members")

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "imported-members": True,
}

# Mock imports for autodoc (avoid importing heavy dependencies)
autodoc_mock_imports = [
    "numpy",
    "jax",
    "jaxlib",
    "torch",
    "tensorrt",
    "mlir_tensorrt",
    "h5py",
    "scipy",
    "flax",
    "optax",
    "orbax",
    "ran.trt_plugins.dmrs",
    "ran.trt_plugins.fft",
    "ran.trt_plugins.cholesky_factor_inv",
]

# Autosummary settings
autosummary_generate = True

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Source file suffixes
# myst-parser handles .md, Sphinx handles .rst
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
