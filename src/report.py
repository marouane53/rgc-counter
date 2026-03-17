# src/report.py

from __future__ import annotations

import datetime
import os
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import pandas as pd
from jinja2 import Template

_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>retinal-phenotyper Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
h1 { margin-bottom: 0; }
h2 { margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
td, th { border: 1px solid #ddd; padding: 6px; font-size: 14px; }
small { color: #777; }
img { max-width: 100%; margin: 10px 0; border: 1px solid #ccc; }
.code { background: #f7f7f7; padding: 12px; font-family: Consolas, monospace; font-size: 13px; }
ul { padding-left: 18px; }
</style>
</head>
<body>
<h1>retinal-phenotyper Report</h1>
<small>Generated at {{ timestamp }}</small>

<h2>Run info</h2>
<div class="code"><pre>{{ run_info }}</pre></div>

{% if summary_table is not none %}
<h2>Summary</h2>
{{ summary_table | safe }}
{% endif %}

{% if tables %}
<h2>Tables</h2>
{% for section in tables %}
<h3>{{ section.title }}</h3>
{{ section.html | safe }}
{% endfor %}
{% endif %}

{% if assets %}
<h2>Artifacts</h2>
<ul>
{% for title, relpath in assets %}
<li><a href="{{ relpath }}">{{ title }}</a></li>
{% endfor %}
</ul>
{% endif %}

{% if images %}
<h2>Visuals</h2>
{% for title, relpath in images %}
<h3>{{ title }}</h3>
<img src="{{ relpath }}" alt="{{ title }}">
{% endfor %}
{% endif %}

{% if methods_appendix %}
<h2>Methods Appendix</h2>
<div class="code"><pre>{{ methods_appendix }}</pre></div>
{% endif %}

{% if notes %}
<h2>Notes</h2>
<p>{{ notes }}</p>
{% endif %}

</body>
</html>
"""


class _ReportReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.references.append(value)


def _is_relative_report_reference(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return False
    if not parsed.path or parsed.path.startswith("/") or parsed.path.startswith("#"):
        return False
    return True


def collect_report_relative_references(report_path: str | Path) -> list[str]:
    report_path = Path(report_path)
    parser = _ReportReferenceParser()
    parser.feed(report_path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    refs: list[str] = []
    for ref in parser.references:
        if not _is_relative_report_reference(ref):
            continue
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def find_missing_report_references(report_path: str | Path) -> list[str]:
    report_path = Path(report_path)
    source_root = report_path.parent.resolve()
    missing: list[str] = []
    for ref in collect_report_relative_references(report_path):
        parsed = urlsplit(ref)
        resolved = (source_root / parsed.path).resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError:
            missing.append(ref)
            continue
        if not resolved.exists():
            missing.append(ref)
    return missing


def copy_report_bundle(report_path: str | Path, destination_dir: str | Path) -> Path:
    report_path = Path(report_path).resolve()
    source_root = report_path.parent
    destination_dir = Path(destination_dir)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_report = destination_dir / report_path.name
    shutil.copy2(report_path, destination_report)

    for ref in collect_report_relative_references(report_path):
        parsed = urlsplit(ref)
        source_asset = (source_root / parsed.path).resolve()
        try:
            relative_asset = source_asset.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"Report reference escapes report root: {ref}") from exc
        if not source_asset.exists():
            raise FileNotFoundError(f"Missing report asset referenced by {report_path}: {ref}")
        destination_asset = destination_dir / relative_asset
        destination_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, destination_asset)

    return destination_report


def write_html_report(
    output_dir: str,
    run_info: Dict[str, Any],
    results_rows: List[Dict[str, Any]],
    images: Optional[List[tuple[str, str]]] = None,
    notes: str = "",
    tables: Optional[List[Dict[str, str]]] = None,
    methods_appendix: str = "",
    assets: Optional[List[tuple[str, str]]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    run_text = "\n".join([f"{k}: {v}" for k, v in run_info.items()])
    df = pd.DataFrame(results_rows)
    summary_html = df.to_html(index=False) if not df.empty else None

    html = Template(_HTML_TEMPLATE).render(
        timestamp=timestamp,
        run_info=run_text,
        summary_table=summary_html,
        tables=tables or [],
        images=images or [],
        assets=assets or [],
        methods_appendix=methods_appendix,
        notes=notes,
    )

    out_path = os.path.join(output_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return out_path
