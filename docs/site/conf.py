"""Sphinx configuration for the impact-adjoint documentation site.

Stack chosen to match the Tesseract projects' own docs (Sphinx + Furo + MyST)
so the pages sit naturally alongside docs.pasteurlabs.ai.
"""

from datetime import date
from pathlib import Path

here = Path(__file__).parent.resolve()

project = "impact-adjoint"
copyright = f"{date.today().year}, Harsh Singh"
author = "Harsh Singh"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

numfig = True
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "impact-adjoint"
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/singhharsh1708/impact-adjoint/",
    "source_branch": "main",
    "source_directory": "docs/site/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/singhharsh1708/impact-adjoint",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
                "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01"
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 "
                "1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 "
                "2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 "
                '0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
html_css_files = ["custom.css"]

ogp_site_url = "https://impact-adjoint.vercel.app/"
ogp_site_name = "impact-adjoint"
ogp_type = "article"
ogp_description_length = 200
ogp_social_cards = {"line_color": "#2a78d6"}

# figures and generated tables live one level up; Sphinx copies what is linked
html_extra_path = []
suppress_warnings = ["myst.header"]

mermaid_init_js = """
(() => {
  const attr = document.body.dataset.theme;
  const dark =
    attr === "dark" ||
    (attr !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  mermaid.initialize({
    startOnLoad: true,
    theme: "base",
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif",
    themeVariables: dark
      ? {
          background: "#131416",
          primaryColor: "#1a1b1e",
          primaryTextColor: "#e8e6e3",
          primaryBorderColor: "#3a3b40",
          lineColor: "#8a8883",
          fontSize: "14px",
        }
      : {
          background: "#fcfcfb",
          primaryColor: "#f4f4f1",
          primaryTextColor: "#16161a",
          primaryBorderColor: "#d8d7d1",
          lineColor: "#898781",
          fontSize: "14px",
        },
  });
})();
"""
