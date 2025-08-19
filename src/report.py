# src/report.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import datetime
import pandas as pd
from jinja2 import Template

_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RGC Counter Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
h1 { margin-bottom: 0; }
h2 { margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
td, th { border: 1px solid #ddd; padding: 6px; font-size: 14px; }
small { color: #777; }
img { max-width: 100%; margin: 10px 0; border: 1px solid #ccc; }
.code { background: #f7f7f7; padding: 12px; font-family: Consolas, monospace; font-size: 13px; }
</style>
</head>
<body>
<h1>RGC Counter Report</h1>
<small>Generated at {{ timestamp }}</small>

<h2>Run info</h2>
<div class="code"><pre>{{ run_info }}</pre></div>

{% if summary_table is not none %}
<h2>Summary</h2>
{{ summary_table | safe }}
{% endif %}

{% if images %}
<h2>Visuals</h2>
{% for title, relpath in images %}
<h3>{{ title }}</h3>
<img src="{{ relpath }}" alt="{{ title }}">
{% endfor %}
{% endif %}

{% if notes %}
<h2>Notes</h2>
<p>{{ notes }}</p>
{% endif %}

</body>
</html>
"""

def write_html_report(output_dir: str,
                      run_info: Dict[str, Any],
                      results_rows: List[Dict[str, Any]],
                      images: Optional[List[tuple]] = None,
                      notes: str = "") -> str:
    """
    Write a self-contained HTML report that references saved images in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    run_text = "\n".join([f"{k}: {v}" for k, v in run_info.items()])
    df = pd.DataFrame(results_rows)
    summary_html = df.to_html(index=False) if not df.empty else None

    html = Template(_HTML_TEMPLATE).render(
        timestamp=timestamp,
        run_info=run_text,
        summary_table=summary_html,
        images=images or [],
        notes=notes
    )

    out_path = os.path.join(output_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

