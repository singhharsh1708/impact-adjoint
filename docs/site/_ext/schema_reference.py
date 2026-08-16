"""Generate component reference tables from the Tesseract schemas.

The reference page used to carry hand-written tables, which is exactly the
drift the Results page exists to prevent: a schema change and a docs change
are two separate acts of memory. These tables are read from the schema source
instead, so they cannot disagree with it.

The schemas are parsed with `ast` rather than imported. Importing
contact_sim's tesseract_api pulls in juliacall and includes the solver, which
would bootstrap a Julia depot inside the docs build; the annotations carry
everything the table needs without running any of that.

An annotation like `Differentiable[Array[(None,), Float64]]` yields the
differentiability flag, the container, the shape and the dtype. Units and
valid ranges live in each field's `description`, where the schema states
them, and are reproduced verbatim.
"""

import ast
import re
from pathlib import Path

import docutils.statemachine
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)


def _unparse(node):
    return ast.unparse(node) if node is not None else ""


def _describe(annotation: str) -> tuple[str, str, str]:
    """(dtype, shape, differentiable) from a Tesseract annotation string."""
    diff = "yes" if annotation.startswith("Differentiable[") else "no"
    inner = annotation
    if diff == "yes":
        inner = annotation[len("Differentiable[") : -1]
    m = re.match(r"Array\[\s*\((.*?)\)\s*,\s*(\w+)\s*\]", inner)
    if m:
        dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
        shape = "(" + ", ".join("n" if d == "None" else d for d in dims) + ")"
        return m.group(2), shape, diff
    return inner, "scalar", diff


def _field_meta(value) -> tuple[str, str]:
    """(description, default) out of a `Field(...)` call."""
    desc = default = ""
    if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "Field":
        for kw in value.keywords:
            if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                desc = kw.value.value
            elif kw.arg == "default":
                default = _unparse(kw.value)
        if value.args and isinstance(value.args[0], ast.Constant):
            default = repr(value.args[0].value)
    elif value is not None:
        default = _unparse(value)
    return " ".join(desc.split()), default


def parse_schema(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in (
            "InputSchema",
            "OutputSchema",
        ):
            continue
        fields = []
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(
                item.target, ast.Name
            ):
                continue
            dtype, shape, diff = _describe(_unparse(item.annotation))
            desc, default = _field_meta(item.value)
            fields.append(
                {
                    "name": item.target.id,
                    "dtype": dtype,
                    "shape": shape,
                    "diff": diff,
                    "default": default,
                    "desc": desc,
                }
            )
        out[node.name] = fields
    return out


class SchemaTable(Directive):
    """`.. schema-table:: <path-from-repo-root> <InputSchema|OutputSchema>`"""

    required_arguments = 2
    has_content = False

    def run(self):
        rel, which = self.arguments
        root = Path(self.state.document.settings.env.srcdir).parent.parent
        path = root / rel
        if not path.exists():
            logger.warning("schema-table: %s not found", rel)
            return []
        fields = parse_schema(path).get(which, [])
        if not fields:
            logger.warning("schema-table: no %s in %s", which, rel)
            return []

        lines = [
            "```{list-table}",
            ":header-rows: 1",
            ":widths: 16 12 10 8 54",
            "",
            "* - Field",
            "  - Type",
            "  - Shape",
            "  - Diff.",
            "  - Description",
        ]
        for f in fields:
            desc = f["desc"] or "-"
            if f["default"]:
                desc += f" Default `{f['default']}`."
            lines += [
                f"* - `{f['name']}`",
                f"  - {f['dtype']}",
                f"  - {f['shape']}",
                f"  - {f['diff']}",
                f"  - {desc}",
            ]
        lines.append("```")

        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(
            docutils.statemachine.StringList(lines), self.content_offset, node
        )
        return node.children


def setup(app):
    app.add_directive("schema-table", SchemaTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
