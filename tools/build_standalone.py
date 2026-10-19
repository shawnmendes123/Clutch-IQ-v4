"""Builds clutchiq.html: a single self-contained file (CSS + JS inlined) that
runs the whole app offline with the in-browser engine — double-click to open.
    python tools/build_standalone.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    css = open(os.path.join(ROOT, "assets", "app.css"), encoding="utf-8").read()
    html = html.replace('<link rel="stylesheet" href="assets/app.css">', "<style>\n" + css + "\n</style>")

    def inline(m):
        src = open(os.path.join(ROOT, m.group(1)), encoding="utf-8").read().replace("</script>", "<\\/script>")
        return "<script>\n" + src + "\n</script>"

    html = re.sub(r'<script src="(assets/[^"]+)"></script>', inline, html)
    out = os.path.join(ROOT, "clutchiq.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
