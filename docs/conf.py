project = "spicyr"
author = "Ellis Patrick, Sadiq Dohadwalla, Elijah Willie"
copyright = "2026, the spicyR authors"
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx_autodoc_typehints"]
html_theme = "sphinx_book_theme"
html_title = "spicyr"
html_theme_options = {"repository_url": "https://github.com/SydneyBioX/spicyr-py", "use_repository_button": True,
                      "show_toc_level": 2}
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
nb_execution_mode = "force"     # the tutorial runs at every build, like an R vignette
nb_execution_timeout = 600
nb_execution_raise_on_error = True
myst_enable_extensions = ["dollarmath", "colon_fence"]
napoleon_numpy_docstring = True
suppress_warnings = ["mystnb.unknown_mime_type"]
