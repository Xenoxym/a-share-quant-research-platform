from __future__ import annotations
import json
from pathlib import Path
from .artifacts import jsonable


def render_page(boot):
    assets = Path(__file__).with_name("assets")
    payload = json.dumps(jsonable(boot), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    script = "\n".join((assets / name).read_text(encoding="utf-8") for name in ["chart.js", "foundations.js", "regimes.js", "tasks.js", "controller.js", "ml.js", "app.js"])
    return (assets / "index.html").read_text(encoding="utf-8").replace("/*STYLE*/", (assets / "style.css").read_text(encoding="utf-8")).replace("/*BOOT*/", payload).replace("/*SCRIPT*/", script)
