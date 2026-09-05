"""Node help text.

The prose for each node lives in `docs/nodes/<NodeName>.md`, one Markdown file per entry in
NODE_CLASS_MAPPINGS. This module loads those files, converts them to the small HTML dialect the
help popup (web/js/help_popup.js) expects, and attaches the result to each node class so ComfyUI
serves it in /object_info.

Supported Markdown (a deliberately small subset - the popup only styles these):

    # Title                 the node's display title (first line)
    first paragraph         the short description, used for canvas tooltips
    plain lines             paragraphs
    - item                  bullets (the "-" is kept as literal text)
    two spaces per level     indents a paragraph or bullet by one level
    `code`                  inline code styling
    ## Section              a collapsible section holding everything below it
    ## Section [collapsed]  the same, but collapsed when the popup opens
"""

import re
from pathlib import Path

from ..utils.logger import logger

# docs/nodes/, resolved relative to this file so it works wherever ComfyUI is installed
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "nodes"

INDENT_PX = 20  # px added per indent level
SPACES_PER_INDENT = 2  # leading spaces in the Markdown that make up one indent level


def process_highlights(text):
    """Convert `highlighted` parts to code style that works in both light and dark themes"""
    pattern = r'`([^`]+)`'
    # Theme-agnostic styling:
    return re.sub(pattern, r'<code style="border: 1px solid #666; border-radius: 3px; padding: 0px 1px; font-family: monospace; display: inline-block;">\1</code>', text)


def _short_desc(text):
    """The short description, tagged with the ID help_popup.js reads to build node tooltips"""
    return f'<div id="Y7_shortdesc" style="margin-bottom: 15px;">{text}</div>'


def _paragraph(text, indent_level=0, font_size="12px"):
    """A line of body text, optionally indented"""
    indent_px = indent_level * INDENT_PX
    processed_text = process_highlights(text)
    return f'<div style="margin-bottom: 8px; margin-left: {indent_px}px; font-size: {font_size};">{processed_text}</div>'


def _section(name, blocks, collapsed=False):
    """A collapsible section; the [-] toggle and its behaviour come from help_popup.js"""
    collapse_flag = ' Y7Nodes_precollapse' if collapsed else ''
    inner = ''.join(f'<div style="font-size: 1em">{block}</div>' for block in blocks)
    return (
        f'<div Y7Nodes_title="{name}" style="display: flex; font-size: 0.8em" class="Y7Nodes_collapse{collapse_flag}">'
        f'<div style="color: #AAA; height: 1.5em;">[<span style="font-family: monospace">-</span>]</div>'
        f'<div style="width: 100%">{name}: {inner}</div>'
        f'</div>'
    )


def md_to_html(markdown):
    """Convert one node's Markdown help file into the HTML the help popup renders.

    Every non-blank line becomes one block; blank lines are only separators. Blocks that appear
    after a `## Section` heading are nested inside that section.
    """
    title = ''
    top_level = []  # blocks before the first ## section
    sections = []  # (name, collapsed, [blocks]) in file order
    seen_body = False  # the first body line is the short description

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("## "):
            name = line[3:].strip()
            collapsed = name.endswith("[collapsed]")
            if collapsed:
                name = name[:-len("[collapsed]")].strip()
            sections.append((name, collapsed, []))
            continue

        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue

        indent_level = (len(raw_line) - len(raw_line.lstrip(" "))) // SPACES_PER_INDENT
        if not seen_body:
            block = _short_desc(process_highlights(line))
            seen_body = True
        else:
            block = _paragraph(line, indent_level)

        if sections:
            sections[-1][2].append(block)
        else:
            top_level.append(block)

    html = title
    for block in top_level:
        html += f'<div style="font-size: 0.8em">{block}</div>'
    for name, collapsed, blocks in sections:
        html += _section(name, blocks, collapsed)
    return html


def load_docs():
    """Read docs/nodes/*.md, returning {node name: help HTML}"""
    docs = {}
    if not DOCS_DIR.is_dir():
        logger.warning(f"Node documentation directory not found: {DOCS_DIR}")
        return docs

    for path in sorted(DOCS_DIR.glob("*.md")):
        try:
            docs[path.stem] = md_to_html(path.read_text(encoding="utf-8"))
        except OSError as e:
            logger.warning(f"Could not read node documentation from {path.name}: {e}")
    return docs


def _apply_v3_description(node_cls, html):
    """Attach documentation HTML to a V3 (io.ComfyNode) node class.

    V3 nodes don't serve the DESCRIPTION class attribute: ComfyUI builds their node info from a
    fresh define_schema() call (Schema.get_v1_info uses schema.description), and DESCRIPTION itself
    is a classproperty backed by _DESCRIPTION. Assigning DESCRIPTION on such a class silently
    shadows the classproperty and is then ignored by the frontend, so patch the schema on its way
    out of define_schema instead.
    """
    node_cls._y7_doc_html = html
    node_cls._DESCRIPTION = html  # so cls.DESCRIPTION reads back the docs too

    if "_y7_doc_patched" in node_cls.__dict__:
        return  # already wrapped; the refreshed _y7_doc_html above is all that's needed

    original_define_schema = node_cls.define_schema

    def define_schema(cls):
        schema = original_define_schema()
        schema.description = cls._y7_doc_html
        return schema

    node_cls.define_schema = classmethod(define_schema)
    node_cls._y7_doc_patched = True


def format_descriptions(nodes):
    """Applies HTML documentation to node classes"""
    docs = load_docs()

    applied = 0
    for name, html in docs.items():
        if name not in nodes:
            logger.warning(f"docs/nodes/{name}.md has no matching node in NODE_CLASS_MAPPINGS")
            continue

        if hasattr(nodes[name], "define_schema"):
            _apply_v3_description(nodes[name], html)
        else:
            nodes[name].DESCRIPTION = html
            # Also set a direct description property for easier access
            nodes[name].description = html
        applied += 1

    undocumented_nodes = [name for name in nodes if name not in docs]
    if undocumented_nodes:
        logger.warning(f"Some nodes have not been documented: {undocumented_nodes}")

    logger.info(f"Applied documentation to {applied} of {len(nodes)} nodes")

    # Return the number of descriptions applied for confirmation
    return applied
