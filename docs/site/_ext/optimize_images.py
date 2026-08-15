"""Post-process built HTML: WebP alongside PNG, lazy loading, intrinsic sizes.

The MyST sources stay plain `{image}` directives pointing at the PNGs, because
those same PNGs are what the README renders on GitHub. This runs after the
build and rewrites each figure into a <picture> that prefers WebP and falls
back to the original PNG, so nothing is lost on a browser that cannot decode
WebP.

Width and height come from the file itself, which lets the browser reserve the
box before the bytes arrive and removes the layout shift. The first figure on
a page keeps eager loading, since it may be the largest contentful paint;
everything below it is deferred.
"""

import re
from pathlib import Path

from PIL import Image
from sphinx.util import logging

logger = logging.getLogger(__name__)

IMG_RE = re.compile(r'<img\s+([^>]*?)src="(_images/([^"]+\.png))"([^>]*?)/?>', re.I)
ATTR_RE = re.compile(r'(\w[\w-]*)\s*=\s*"([^"]*)"')


def _webp_for(png: Path) -> Path | None:
    """Write a sibling .webp, returning it only when it actually saves bytes."""
    webp = png.with_suffix(".webp")
    if not webp.exists():
        with Image.open(png) as im:
            im.convert("RGB").save(webp, "WEBP", quality=82, method=6)
    if webp.stat().st_size >= png.stat().st_size:
        webp.unlink()
        return None
    return webp


def _rewrite(html: str, images_dir: Path) -> tuple[str, int]:
    seen = {"n": 0}

    def repl(m):
        before, src, name, after = m.groups()
        png = images_dir / name
        if not png.exists():
            return m.group(0)
        attrs = dict(ATTR_RE.findall(before + " " + after))
        try:
            with Image.open(png) as im:
                w, h = im.size
        except OSError:
            return m.group(0)
        attrs.setdefault("width", str(w))
        attrs.setdefault("height", str(h))
        attrs["decoding"] = "async"
        seen["n"] += 1
        if seen["n"] > 1:
            attrs["loading"] = "lazy"
        attrs["src"] = src
        img = "<img " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + " />"
        webp = _webp_for(png)
        if webp is None:
            return img
        return (
            f'<picture><source srcset="_images/{webp.name}" type="image/webp" />'
            f"{img}</picture>"
        )

    return IMG_RE.sub(repl, html), seen["n"]


def on_build_finished(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    out = Path(app.outdir)
    images_dir = out / "_images"
    if not images_dir.is_dir():
        return
    total = 0
    for page in out.glob("*.html"):
        html = page.read_text(encoding="utf-8")
        new, n = _rewrite(html, images_dir)
        if n:
            page.write_text(new, encoding="utf-8")
            total += n
    saved = sum(p.stat().st_size for p in images_dir.glob("*.png")) - sum(
        p.stat().st_size for p in images_dir.glob("*.webp")
    )
    logger.info(
        "optimize_images: rewrote %d <img>, WebP saves %.0f KB", total, saved / 1024
    )


def setup(app):
    app.connect("build-finished", on_build_finished)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
