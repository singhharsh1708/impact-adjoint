"""Head-tag corrections for a site served on clean URLs.

Vercel serves this site with `cleanUrls`, so `/getting-started` is canonical
and `/getting-started.html` is not. Sphinx and the OpenGraph extension both
emit the `.html` form, and Furo and Sphinx each emit their own viewport tag.
Rather than fork the theme templates for four small fixes, the built HTML is
corrected here:

  * collapse the duplicate <meta name="viewport">
  * rewrite canonical and og:url to the clean form
  * add the Twitter tags that were falling back to OpenGraph
  * add JSON-LD describing the artifact

The sitemap is rewritten to the same clean URLs so it agrees with canonical.
"""

import json
import re
from pathlib import Path

from sphinx.util import logging

logger = logging.getLogger(__name__)

VIEWPORT_RE = re.compile(r'\s*<meta name="viewport"[^>]*>', re.I)
VIEWPORT_TAG = '\n    <meta name="viewport" content="width=device-width, initial-scale=1" />'


def _clean(url: str) -> str:
    """`.../index.html` -> `.../`, `.../page.html` -> `.../page`."""
    if url.endswith("/index.html"):
        return url[: -len("index.html")]
    if url.endswith(".html"):
        return url[: -len(".html")]
    return url


def _structured_data(app) -> str:
    meta = app.config.site_meta
    data = {
        "@context": "https://schema.org",
        "@type": "SoftwareSourceCode",
        "name": app.config.project,
        "description": meta["description"],
        "url": app.config.html_baseurl,
        "codeRepository": meta["repository"],
        "programmingLanguage": [{"@type": "ComputerLanguage", "name": n}
                                for n in meta["languages"]],
        "license": meta["license_url"],
        "author": {
            "@type": "Person",
            "name": app.config.author,
            "url": meta["author_url"],
        },
        "version": app.config.release,
    }
    return (
        '<script type="application/ld+json">'
        + json.dumps(data, separators=(",", ":"))
        + "</script>"
    )


def _twitter(app, html: str) -> str:
    """Mirror the page's own OpenGraph card; its filename is content-hashed."""
    meta = app.config.site_meta
    m = re.search(r'<meta property="og:image" content="([^"]+)"', html)
    tags = [
        f'<meta name="twitter:title" content="{app.config.project}" />',
        f'<meta name="twitter:description" content="{meta["description"]}" />',
    ]
    if m:
        tags.append(f'<meta name="twitter:image" content="{m.group(1)}" />')
    return "".join(tags)


def _provenance(app) -> str:
    """A footer line naming the commit the site was built from."""
    meta = app.config.site_meta
    sha, built = meta.get("commit"), meta.get("built")
    if not sha:
        return ""
    short = sha[:7]
    href = f'{meta["repository"]}/commit/{sha}'
    return (
        '<div class="ia-provenance">Built from '
        f'<a class="muted-link" href="{href}"><code>{short}</code></a> '
        f"on {built}.</div>"
    )


LEFT_DETAILS_END = re.compile(r'(</div>\s*<div class="right-details">)')


def on_build_finished(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    out = Path(app.outdir)
    base = app.config.html_baseurl.rstrip("/") + "/"
    fixed = 0

    for page in out.rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        original = html

        # one viewport, not two
        if len(VIEWPORT_RE.findall(html)) > 1:
            html = VIEWPORT_RE.sub("", html, count=len(VIEWPORT_RE.findall(html)))
            html = html.replace("</title>", "</title>" + VIEWPORT_TAG, 1)

        # canonical and og:url on the URL the site actually serves
        rel = page.relative_to(out).as_posix()
        canonical = _clean(base + rel)
        html = re.sub(
            r'<link rel="canonical"[^>]*>',
            f'<link rel="canonical" href="{canonical}" />',
            html,
        )
        html = re.sub(
            r'(<meta property="og:url" content=")([^"]*)(")',
            lambda m: m.group(1) + _clean(m.group(2)) + m.group(3),
            html,
        )

        if "twitter:title" not in html:
            html = html.replace("</head>", _twitter(app, html) + "</head>", 1)
        if "application/ld+json" not in html and rel == "index.html":
            html = html.replace(
                "</head>", _structured_data(app) + "</head>", 1
            )

        # Furo's two <aside> landmarks are indistinguishable to a screen
        # reader, and its page-action links sit outside every landmark.
        html = html.replace(
            '<aside class="sidebar-drawer">',
            '<aside class="sidebar-drawer" aria-label="Site navigation">',
        )
        html = html.replace(
            '<aside class="toc-drawer">',
            '<aside class="toc-drawer" aria-label="On this page">',
        )
        html = html.replace(
            '<div class="content-icon-container">',
            '<div class="content-icon-container" role="navigation"'
            ' aria-label="Page actions">',
        )

        # Furo emits role="heading" on sidebar captions without aria-level,
        # which the role requires. Supply it rather than dropping the role.
        html = html.replace(
            '<p class="caption" role="heading">',
            '<p class="caption" role="heading" aria-level="2">',
        )

        prov = _provenance(app)
        if prov and "ia-provenance" not in html:
            html = LEFT_DETAILS_END.sub(prov + r"\1", html, count=1)

        if html != original:
            page.write_text(html, encoding="utf-8")
            fixed += 1

    # the sitemap must agree with canonical
    sitemap = out / "sitemap.xml"
    if sitemap.exists():
        xml = sitemap.read_text(encoding="utf-8")
        xml = re.sub(r"<loc>([^<]+)</loc>",
                     lambda m: f"<loc>{_clean(m.group(1))}</loc>", xml)
        sitemap.write_text(xml, encoding="utf-8")

    logger.info("site_meta: corrected head tags on %d pages", fixed)


def setup(app):
    app.add_config_value("site_meta", {}, "html")
    app.connect("build-finished", on_build_finished)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
