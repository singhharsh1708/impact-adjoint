"""Sphinx configuration for the impact-adjoint documentation site.

Stack chosen to match the Tesseract projects' own docs (Sphinx + Furo + MyST)
so the pages sit naturally alongside docs.pasteurlabs.ai.
"""

import json
import sys
from datetime import date
from pathlib import Path

here = Path(__file__).parent.resolve()
sys.path.insert(0, str(here / "_ext"))

project = "impact-adjoint"
copyright = f"{date.today().year}, Harsh Singh"
author = "Harsh Singh"
release = "0.1.1"
version = "0.1.1"

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
    "sphinx_sitemap",
    "optimize_images",
    "site_meta",
    "sweep_widget",
    "schema_reference",
    "figure_source",
    "diffrax_table",
    "stat_cards",
    "artifact_index",
]

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist", "attrs_inline",
                          "substitution"]
myst_heading_anchors = 3
# Flip to False the moment the repository is public; that is the only edit
# needed to remove the notice from every page that carries it.
repo_is_private = False

_private_note = (
    ":::{warning}\n"
    "**The repository is private while the hackathon entry is under review, so "
    "the links and the clone command on this page return 404 to anyone outside "
    "the author's account.** It opens when judging concludes. Everything "
    "described here is in that repository and reproduces from it; until then "
    "the site is the only public record of it.\n"
    ":::"
) if repo_is_private else ""

# Read from the artifact rather than typed, for the same reason the stat cards
# are: prose on this page is not covered by the drift test.
_timing = json.loads((here.parent.parent / "experiments" / "timing.json").read_text())
_checks_walltime = f"about {_timing['total_median_s']:.0f} seconds"

myst_substitutions = {
    "version": version,
    "repo_note": _private_note,
    "checks_walltime": _checks_walltime,
}

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
html_js_files = ["ia_sweep.js"]

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/repo.html",
        "sidebar/scroll-end.html",
    ]
}


def _build_provenance():
    """Commit and build date for the footer, so the site is traceable too."""
    import os
    import subprocess

    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            sha = None
    return sha, date.today().isoformat()


def _repo_stars(url):
    """Star count for the sidebar, or None while the repository is private."""
    import json
    import urllib.request

    try:
        api = url.replace("https://github.com/", "https://api.github.com/repos/")
        with urllib.request.urlopen(api, timeout=4) as r:
            return json.load(r).get("stargazers_count")
    except Exception:
        return None


_commit, _built = _build_provenance()

html_context = {
    "ia_repo_url": "https://github.com/singhharsh1708/impact-adjoint",
    "ia_repo_stars": _repo_stars("https://github.com/singhharsh1708/impact-adjoint"),
}

html_baseurl = "https://impact-adjoint.vercel.app/"
sitemap_url_scheme = "{link}"
sitemap_excludes = ["404.html", "genindex.html", "search.html"]
html_extra_path = ["_extra"]

# consumed by the site_meta extension for Twitter tags and JSON-LD
site_meta = {
    "description": (
        "Exact gradients through impact events, across a Julia and JAX "
        "Tesseract boundary."
    ),
    "repository": "https://github.com/singhharsh1708/impact-adjoint",
    "author_url": "https://github.com/singhharsh1708",
    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
    "languages": ["Julia", "Python"],
    # Sphinx auto-extracts the description from the first content, which now
    # starts with the byline and truncates mid-word on "Version". The index
    # gets an explicit one carrying the claim instead.
    "index_description": (
        "Simulate a bouncing ball in JAX the natural way and jax.grad returns "
        "exactly 0.0, where the true value is +0.09. impact-adjoint supplies "
        "the missing term with the classical saltation matrix, served from a "
        "Julia solver through a Tesseract boundary."
    ),
    "commit": _commit,
    "built": _built,
}

ogp_site_url = "https://impact-adjoint.vercel.app/"
ogp_site_name = "impact-adjoint"
ogp_type = "article"
ogp_description_length = 200
ogp_social_cards = {"line_color": "#2a78d6"}

suppress_warnings = ["myst.header"]

_MERMAID_INIT_JS = """
(() => {
  const attr = document.body.dataset.theme;
  const dark =
    attr === "dark" ||
    (attr !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  mermaid.initialize({
    startOnLoad: true,
    theme: "base",
    // a wrapped subgraph title is drawn over the box edge without this
    flowchart: {
      padding: 16,
      nodeSpacing: 52,
      rankSpacing: 105,
      subGraphTitleMargin: { top: 8, bottom: 12 },
    },
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif",
    themeVariables: dark
      ? {
          background: "#131416",
          edgeLabelBackground: "#131416",
          primaryColor: "#1a1b1e",
          primaryTextColor: "#e8e6e3",
          primaryBorderColor: "#3a3b40",
          lineColor: "#8a8883",
          fontSize: "14px",
        }
      : {
          background: "#fcfcfb",
          edgeLabelBackground: "#fcfcfb",
          primaryColor: "#f4f4f1",
          primaryTextColor: "#16161a",
          primaryBorderColor: "#d8d7d1",
          lineColor: "#898781",
          fontSize: "14px",
        },
  });
})();
"""

# sphinxcontrib-mermaid 2.1.0 registers `mermaid_init_config`, not
# `mermaid_init_js`. The theme block above was assigned to a name Sphinx
# silently ignores, so none of it ever reached a page. Pass the parts that are
# static as config, and keep the theme-following logic in a small script.
mermaid_init_config = {
    "startOnLoad": True,
    "theme": "base",
    "flowchart": {
        "padding": 16,
        "nodeSpacing": 52,
        "rankSpacing": 105,
        "subGraphTitleMargin": {"top": 8, "bottom": 12},
    },
    # no fontFamily override: mermaid measures label widths against the font it
    # believes it is using, and naming a webfont it cannot measure clips every
    # node label. The page CSS styles the rendered SVG text instead.
}
