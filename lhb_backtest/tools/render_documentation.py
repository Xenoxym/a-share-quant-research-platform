"""Render trusted repository Markdown with the installed markdown-it-py library."""

import argparse
import re
from pathlib import Path

from markdown_it import MarkdownIt

DOCS = Path(__file__).resolve().parents[1] / "docs"
STYLE = """body{margin:0;background:#f3f6fa;color:#20304a;font:17px/1.8 system-ui}
main{max-width:1080px;margin:32px auto;padding:32px;background:white;border-radius:16px}
h1,h2{line-height:1.4}h2{margin-top:38px;color:#17436b}a{color:#16639c}
table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #d7e0eb;
padding:10px;text-align:left;vertical-align:top}th{background:#edf3fa}
pre{padding:18px;background:#edf3fa;overflow:auto;border-radius:8px}
code{font-size:14px}img{max-width:100%;height:auto}blockquote{border-left:4px solid #4d83b8;
margin:18px 0;padding:8px 20px;background:#f2f6fc}
@media(max-width:700px){main{margin:0;padding:18px}table{display:block;overflow:auto}}"""


def render(path):
    path = Path(path).resolve()
    if not path.is_relative_to(DOCS) or path.suffix != ".md":
        raise ValueError("Only Markdown under this repository's docs may be rendered")
    md = path.read_text(encoding="utf-8")
    diagram = 0

    def replace_diagram(match):
        nonlocal diagram
        diagram += 1
        name = (
            "PROJECT_ARCHITECTURE_SIMPLE.svg" if diagram == 1 else "PROJECT_ARCHITECTURE_DETAIL.svg"
        )
        return f"![项目架构图]({name})\n"

    md = re.sub(r"```mermaid\n.*?\n```", replace_diagram, md, flags=re.S)
    body = MarkdownIt("default", {"html": False}).render(md)
    output = path.with_suffix(".html")
    title = path.stem
    output.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{STYLE}</style><main>{body}</main></html>",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    a = p.parse_args()
    for file in a.files:
        print(render(file))
