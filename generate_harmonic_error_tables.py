#!/usr/bin/env python3
"""Generate one harmonic-error CSV and an interactive Plotly HTML report
from RhythmMamba evaluation folders.

The script reads every FAILURE_TYPE_SUMMARY.csv found below the supplied
result roots.  It does not rerun inference and does not modify result files.

The HTML report offers three comparison modes via dropdown:
  1. Checkpoint comparison  — grouped by checkpoint, filtered by dataset
  2. Dataset comparison     — grouped by dataset,    filtered by checkpoint
  3. Table with heatmap     — per-checkpoint summary across all datasets

Plotly.js is embedded inline so the HTML works fully offline.
"""

import csv
import json
import re
from pathlib import Path


SCRIPT_VERSION = "2026-09-01-fixed-html-versioning"


# ---------------------------------------------------------------------------
# Column and name mappings
# ---------------------------------------------------------------------------

FAILURE_COLUMNS = {
    "correct": "correct_count",
    "super_harmonic_1p5x": "harmonic_1p5x_count",
    "super_harmonic_2x": "harmonic_2x_count",
    "sub_harmonic_half": "harmonic_0p5x_count",
    "large_error": "other_large_error_count",
}

PROTOCOL_NAMES = {
    "official_mamba": "Official",
    "old": "Old",
    "prism": "PRISM",
}

DATASET_NAMES = {
    "PURE": "PURE",
    "UBFC": "UBFC-rPPG",
    "TOKYOTECH": "TokyoTech",
    "BH": "BH-rPPG",
    "UBFC_PHYS": "UBFC-PHYS",
    "COHFACE": "COHFACE",
}

KNOWN_CHECKPOINT_NAMES = {
    "PURE_CHECKPOINT": "Our PURE intra (epoch 29)",
    "UBFC_CHECKPOINT": "Our UBFC intra (epoch 29)",
    "OFFICIAL_PURE_CHECKPOINT": "Released PURE cross",
    "OFFICIAL_UBFC_CHECKPOINT": "Released UBFC cross",
    "PURE_CROSS_MATCHED": "Our PURE cross-matched",
    "UBFC_CROSS_MATCHED": "Our UBFC cross-matched (seed 100)",
    "UBFC_CROSS_MATCHED_SEED101": "Our UBFC cross-matched (seed 101)",
    "UBFC_CROSS_MATCHED_SEED102": "Our UBFC cross-matched (seed 102)",
}

OUTPUT_COLUMNS = [
    "evaluation_run",
    "checkpoint",
    "dataset",
    "protocol",
    "evaluation_unit",
    "total_count",
    "correct_count",
    "harmonic_1p5x_count",
    "harmonic_2x_count",
    "harmonic_0p5x_count",
    "other_large_error_count",
]

# Resolve paths from this script, so execution works from any current folder.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "error_analysis"
CSV_PREFIX = "HARMONIC_ERROR_COUNTS_ALL_SETUPS"
HTML_PREFIX = "HARMONIC_ERROR_ANALYSIS"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ---------------------------------------------------------------------------
# CSV collection helpers
# ---------------------------------------------------------------------------

def checkpoint_name(folder_name):
    """Map a folder name to a human-readable checkpoint label."""
    return KNOWN_CHECKPOINT_NAMES.get(folder_name, folder_name)


def discover_evaluation_roots(results_root):
    """Return every present/future ``evaluation_protocols*`` directory."""
    return sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and path.name.startswith("evaluation_protocols")
    )


def read_failure_counts(csv_path):
    """Return a dict of failure-type counts parsed from one summary CSV."""
    counts = {name: 0 for name in FAILURE_COLUMNS.values()}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            output_name = FAILURE_COLUMNS.get(row["failure_type"].strip())
            if output_name:
                counts[output_name] = int(float(row["count"]))
    return counts


def parse_result_path(results_root, csv_path):
    """Extract evaluation metadata from the directory structure.

    Expected layout:
        <results_root>/.../<checkpoint_folder>/Eval_On_<dataset>/<protocol>/...
    Returns None when the path does not match.
    """
    relative = csv_path.relative_to(results_root)
    parts = relative.parts
    try:
        eval_index = next(
            i for i, part in enumerate(parts) if part.startswith("Eval_On_")
        )
    except StopIteration:
        return None

    if eval_index == 0 or eval_index + 1 >= len(parts):
        return None

    evaluation_run = "/".join(parts[: eval_index - 1]) or "results"
    checkpoint_folder = parts[eval_index - 1]
    dataset_key = parts[eval_index].replace("Eval_On_", "", 1)
    protocol_folder = parts[eval_index + 1]
    protocol = PROTOCOL_NAMES.get(protocol_folder)
    if protocol is None:
        return None

    return (
        evaluation_run,
        checkpoint_folder,
        dataset_key,
        protocol_folder,
        protocol,
    )


def collect_rows(results_root):
    """Walk *results_root* and return a sorted list of row dicts."""
    rows = []
    seen_files = set()

    if not results_root.exists():
        raise SystemExit("Results folder was not found: " + str(results_root))

    evaluation_roots = discover_evaluation_roots(results_root)
    if not evaluation_roots:
        raise SystemExit(
            "No evaluation_protocols* folders were found below: "
            + str(results_root)
        )

    for csv_path in (
        path
        for evaluation_root in evaluation_roots
        for path in evaluation_root.rglob("FAILURE_TYPE_SUMMARY.csv")
    ):
        resolved = csv_path.resolve()
        if resolved in seen_files:
            continue
        seen_files.add(resolved)

        parsed = parse_result_path(results_root, csv_path)
        if parsed is None:
            print("Skipped unrecognized path:", csv_path)
            continue

        (
            evaluation_run,
            checkpoint_folder,
            dataset_key,
            protocol_folder,
            protocol,
        ) = parsed
        counts = read_failure_counts(csv_path)
        total = sum(counts.values())

        row = {
            "evaluation_run": evaluation_run,
            "checkpoint": checkpoint_name(checkpoint_folder),
            "dataset": DATASET_NAMES.get(dataset_key, dataset_key),
            "protocol": protocol,
            "evaluation_unit": (
                "recording" if protocol_folder == "official_mamba" else "window"
            ),
            "total_count": total,
            **counts,
        }

        category_sum = sum(row[name] for name in FAILURE_COLUMNS.values())
        if category_sum != total:
            raise ValueError("Category-count mismatch in " + str(csv_path))
        rows.append(row)

    protocol_order = {"Official": 0, "Old": 1, "PRISM": 2}
    rows.sort(
        key=lambda r: (
            r["evaluation_run"],
            r["checkpoint"],
            r["dataset"],
            protocol_order.get(r["protocol"], 99),
        )
    )
    return rows


def write_csv(rows, output_path):
    """Persist *rows* as a CSV file at *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotly.js embedding
# ---------------------------------------------------------------------------

def _get_plotly_js():
    """Obtain the Plotly.js source string for inline embedding.

    Resolution order:
      1. Installed ``plotly`` Python package (``pip install plotly``).
      2. Direct download from CDN via urllib.
    Returns the JavaScript source, or None if both methods fail.
    """
    # --- attempt 1: installed Python package ---
    try:
        from plotly.offline import get_plotlyjs  # type: ignore[import]
        return get_plotlyjs()
    except Exception:
        pass

    # --- attempt 2: download from CDN ---
    try:
        from urllib.request import urlopen
        print("Downloading Plotly.js from CDN …")
        with urlopen(PLOTLY_CDN, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Interactive HTML report generation
# ---------------------------------------------------------------------------

def generate_html_report(rows, output_html_path):
    """Build a self-contained Plotly HTML dashboard from collected data rows.

    Plotly.js is embedded inline when available so the file works offline.
    Three comparison views are available via a primary dropdown:
      Comp 1 – Checkpoint comparison (secondary: choose dataset)
      Comp 2 – Dataset comparison    (secondary: choose checkpoint)
      Comp 3 – Summary table         (secondary: choose checkpoint)
    An evaluation-run dropdown disambiguates rows that share a checkpoint name.
    Bar charts display error-type percentages (count / total × 100).
    """
    data_json = json.dumps(rows, indent=None)

    plotly_js_src = _get_plotly_js()
    if plotly_js_src is not None:
        plotly_script_tag = "<script>\n" + plotly_js_src + "\n</script>"
    else:
        print("WARNING: Could not embed Plotly.js — falling back to CDN link.")
        print("         Install plotly (pip install plotly) for offline use.")
        plotly_script_tag = f'<script src="{PLOTLY_CDN}"></script>'

    html = _html_template(data_json, plotly_script_tag)
    Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_html_path).write_text(html, encoding="utf-8")


def _html_template(data_json, plotly_script_tag):
    """Return the full HTML string with embedded data and Plotly logic."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RhythmMamba — Harmonic Error Analysis</title>
{plotly_script_tag}
<style>
  :root {{
    --bg: #f8f9fa; --card: #ffffff;
    --border: #dee2e6; --text: #212529;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px;
  }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ font-size: 0.9rem; color: #6c757d; margin-bottom: 20px; }}
  .controls {{
    display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
    margin-bottom: 20px; background: var(--card);
    padding: 14px 20px; border-radius: 10px; border: 1px solid var(--border);
  }}
  .controls label {{ font-weight: 600; font-size: 0.85rem; color: #495057; }}
  .controls select {{
    padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
    font-size: 0.9rem; background: #fff; cursor: pointer; min-width: 220px;
  }}
  .plot-row {{
    display: flex; flex-direction: column; gap: 24px; margin-bottom: 20px;
  }}
  .plot-card {{
    width: 100%; background: var(--card); border-radius: 10px;
    border: 1px solid var(--border); padding: 16px;
  }}
  #table-container {{
    background: var(--card); border-radius: 10px;
    border: 1px solid var(--border); padding: 20px;
    overflow-x: auto; display: none;
  }}
  #table-container table {{
    border-collapse: collapse; width: 100%; font-size: 0.85rem;
  }}
  #table-container th, #table-container td {{
    border: 1px solid var(--border); padding: 7px 10px;
    text-align: center; white-space: nowrap;
  }}
  #table-container thead th {{
    background: #343a40; color: #fff; font-weight: 600;
    position: sticky; top: 0;
  }}
  #table-container thead tr:nth-child(2) th {{
    background: #495057; font-weight: 500;
  }}
  #table-container tbody td.ds {{
    font-weight: 600; background: #e9ecef; text-align: left;
  }}
</style>
</head>
<body>

<h1>RhythmMamba — Harmonic Error Analysis</h1>
<p class="subtitle">Interactive comparison across evaluation runs, checkpoints, datasets, and protocols</p>

<div class="controls">
  <label for="sel-run">Run:</label>
  <select id="sel-run"></select>
  <label for="sel-primary">Comparison:</label>
  <select id="sel-primary">
    <option value="comp1">Checkpoint Comparison</option>
    <option value="comp2">Dataset Comparison</option>
    <option value="comp3">Table — Eval Dataset Summary</option>
  </select>
  <label for="sel-secondary" id="lbl-secondary">Dataset:</label>
  <select id="sel-secondary"></select>
</div>

<div class="plot-row" id="plot-row">
  <div class="plot-card"><div id="plot-left" style="width:100%;height:560px;"></div></div>
  <div class="plot-card"><div id="plot-right" style="width:100%;height:560px;"></div></div>
</div>

<div id="table-container"></div>

<script>
// -------------------------------------------------------------------
// Embedded evaluation data (generated at build time)
// -------------------------------------------------------------------
const DATA = {data_json};

// -------------------------------------------------------------------
// Constants — error types plotted (Correct is excluded from bar charts)
// -------------------------------------------------------------------
const ERROR_FIELDS = [
  {{ key: 'harmonic_1p5x_count',     label: '1.5\u00d7' }},
  {{ key: 'harmonic_2x_count',       label: '2\u00d7'   }},
  {{ key: 'harmonic_0p5x_count',     label: '0.5\u00d7' }},
  {{ key: 'other_large_error_count', label: 'Other'  }},
];

// Solid fill colours (used for Old and Official)
const COLORS = {{
  '1.5\u00d7': '#e67e22',
  '2\u00d7':   '#e74c3c',
  '0.5\u00d7': '#8e44ad',
  'Other':     '#7f8c8d',
}};
// Lighter tints (used for PRISM, paired with a diagonal pattern)
const COLORS_LIGHT = {{
  '1.5\u00d7': '#f0b27a',
  '2\u00d7':   '#f1948a',
  '0.5\u00d7': '#bb8fce',
  'Other':     '#b2bec3',
}};

const WINDOW_PROTOCOLS   = ['Old', 'PRISM'];
const RECORDING_PROTOCOL = 'Official';

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------
function unique(arr, field) {{
  return [...new Set(arr.map(r => r[field]))];
}}
function sumField(arr, field) {{
  return arr.reduce((s, r) => s + (r[field] || 0), 0);
}}
// Return DATA rows matching the currently selected evaluation run.
// When "All runs" is active, all rows are returned unfiltered.
function runData() {{
  const v = selRun.value;
  if (v === '__ALL__') return DATA;
  return DATA.filter(r => r.evaluation_run === v);
}}

// -------------------------------------------------------------------
// DOM references
// -------------------------------------------------------------------
const selRun       = document.getElementById('sel-run');
const selPrimary   = document.getElementById('sel-primary');
const selSecondary = document.getElementById('sel-secondary');
const lblSecondary = document.getElementById('lbl-secondary');
const plotRow      = document.getElementById('plot-row');
const tableDiv     = document.getElementById('table-container');

// -------------------------------------------------------------------
// Dropdown population
// -------------------------------------------------------------------
function populateRuns() {{
  selRun.innerHTML = '';
  selRun.add(new Option('All runs', '__ALL__'));
  unique(DATA, 'evaluation_run').forEach(r => {{
    selRun.add(new Option(r, r));
  }});
  populateSecondary();
}}

function populateSecondary() {{
  const mode = selPrimary.value;
  const rd   = runData();
  selSecondary.innerHTML = '';

  if (mode === 'comp1') {{
    lblSecondary.textContent = 'Dataset:';
    unique(rd, 'dataset').forEach(d => {{
      selSecondary.add(new Option(d, d));
    }});
  }} else {{
    lblSecondary.textContent = 'Checkpoint:';
    unique(rd, 'checkpoint').forEach(c => {{
      selSecondary.add(new Option(c, c));
    }});
  }}
  render();
}}

// -------------------------------------------------------------------
// Grouped bar-chart builder — plots PERCENTAGES (count / total × 100)
// -------------------------------------------------------------------
function buildGroupedBar(divId, filtered, groupField, protocols,
                         unitLabel, title) {{
  const groups = unique(filtered, groupField);
  const traces = [];

  protocols.forEach(proto => {{
    const protoRows = filtered.filter(r => r.protocol === proto);
    const isPattern = (proto === 'PRISM');

    ERROR_FIELDS.forEach(et => {{
      const pctVals = [];
      const rawVals = [];

      groups.forEach(g => {{
        const row = protoRows.find(r => r[groupField] === g);
        if (row && row.total_count > 0) {{
          pctVals.push(
            parseFloat(((row[et.key] / row.total_count) * 100).toFixed(2))
          );
          rawVals.push(row[et.key]);
        }} else {{
          pctVals.push(0);
          rawVals.push(0);
        }}
      }});

      const trace = {{
        name: proto + ' \u2014 ' + et.label,
        x: groups,
        y: pctVals,
        customdata: rawVals,
        type: 'bar',
        legendgroup: proto + '_' + et.label,
        text: pctVals.map(v => v > 0 ? v.toFixed(1) + '%' : ''),
        textposition: 'outside',
        textfont: {{ size: 11, color: '#333', weight: 600 }},
        cliponaxis: false,
        marker: {{
          color: isPattern ? COLORS_LIGHT[et.label] : COLORS[et.label],
          line: {{ color: COLORS[et.label], width: 1 }},
        }},
        hovertemplate:
          '<b>' + proto + '</b><br>' +
          '%{{x}}<br>' + et.label +
          ': %{{y:.1f}}%  (count: %{{customdata}})<extra></extra>',
      }};

      if (isPattern) {{
        trace.marker.pattern = {{
          shape: '/', size: 6, solidity: 0.5,
          fgcolor: COLORS[et.label],
        }};
      }}
      traces.push(trace);
    }});
  }});

  // Subtitle showing absolute totals per protocol
  const subtitleParts = protocols.map(proto => {{
    const t = sumField(filtered.filter(r => r.protocol === proto), 'total_count');
    return proto + ': ' + t + ' ' + unitLabel;
  }});

  const layout = {{
    title: {{
      text: title +
            '<br><sup>' + subtitleParts.join(' &nbsp;|&nbsp; ') + '</sup>',
      font: {{ size: 15 }},
    }},
    barmode: 'group',
    xaxis: {{
      title: {{
        text: groupField === 'checkpoint' ? 'Checkpoint' : 'Dataset',
        standoff: 10,
      }},
      tickangle: -25,
      automargin: true,
    }},
    yaxis: {{
      title: {{ text: 'Percentage of Total (' + unitLabel + ')' }},
      autorange: true,
      rangemode: 'tozero',
      ticksuffix: '%',
    }},
    uniformtext: {{ minsize: 7, mode: 'hide' }},
    legend: {{
      orientation: 'h', y: -0.32, x: 0.5, xanchor: 'center',
      font: {{ size: 10 }},
    }},
    margin: {{ t: 70, b: 130, l: 65, r: 20 }},
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#ffffff',
  }};

  Plotly.newPlot(divId, traces, layout, {{ responsive: true }});
}}

// -------------------------------------------------------------------
// Comp 1 — Checkpoint Comparison (filter by dataset)
// -------------------------------------------------------------------
function renderComp1() {{
  const dataset  = selSecondary.value;
  const filtered = runData().filter(r => r.dataset === dataset);

  buildGroupedBar(
    'plot-left',
    filtered.filter(r => WINDOW_PROTOCOLS.includes(r.protocol)),
    'checkpoint', WINDOW_PROTOCOLS, 'windows',
    'Old & PRISM \u2014 ' + dataset
  );
  buildGroupedBar(
    'plot-right',
    filtered.filter(r => r.protocol === RECORDING_PROTOCOL),
    'checkpoint', [RECORDING_PROTOCOL], 'recordings',
    'Official \u2014 ' + dataset
  );
}}

// -------------------------------------------------------------------
// Comp 2 — Dataset Comparison (filter by checkpoint)
// -------------------------------------------------------------------
function renderComp2() {{
  const checkpoint = selSecondary.value;
  const filtered   = runData().filter(r => r.checkpoint === checkpoint);

  buildGroupedBar(
    'plot-left',
    filtered.filter(r => WINDOW_PROTOCOLS.includes(r.protocol)),
    'dataset', WINDOW_PROTOCOLS, 'windows',
    'Old & PRISM \u2014 ' + checkpoint
  );
  buildGroupedBar(
    'plot-right',
    filtered.filter(r => r.protocol === RECORDING_PROTOCOL),
    'dataset', [RECORDING_PROTOCOL], 'recordings',
    'Official \u2014 ' + checkpoint
  );
}}

// -------------------------------------------------------------------
// Comp 3 — Summary table with heatmap colouring
// -------------------------------------------------------------------
function cellBg(value, total, isCorrect) {{
  if (total === 0) return '#f8f9fa';
  const ratio = value / total;
  if (isCorrect) {{
    const g = Math.round(180 + 75 * ratio);
    const r = Math.round(255 - 130 * ratio);
    return 'rgb(' + r + ',' + g + ',140)';
  }}
  if (ratio === 0) return '#f8f9fa';
  const r = Math.round(200 + 55 * Math.min(ratio * 4, 1));
  const g = Math.round(230 - 130 * Math.min(ratio * 4, 1));
  return 'rgb(' + r + ',' + g + ',130)';
}}

function renderComp3() {{
  const checkpoint = selSecondary.value;
  const filtered   = runData().filter(r => r.checkpoint === checkpoint);
  const datasets   = unique(filtered, 'dataset');

  const protocols = [
    {{ key: 'Official', label: 'Official (recordings)' }},
    {{ key: 'Old',      label: 'Old@1bpm (windows)' }},
    {{ key: 'PRISM',    label: 'PRISM (windows)' }},
  ];

  const subCols = ['Total', 'Correct', '1.5\u00d7', '2\u00d7', '0.5\u00d7', 'Other'];
  const subKeys = [
    'total_count', 'correct_count', 'harmonic_1p5x_count',
    'harmonic_2x_count', 'harmonic_0p5x_count', 'other_large_error_count',
  ];

  // --- header ---
  let h = '<table><thead><tr><th rowspan="2">Eval Dataset</th>';
  protocols.forEach(p => {{
    h += '<th colspan="' + subCols.length + '">' + p.label + '</th>';
  }});
  h += '</tr><tr>';
  protocols.forEach(() => {{
    subCols.forEach(sc => {{ h += '<th>' + sc + '</th>'; }});
  }});
  h += '</tr></thead><tbody>';

  // --- body ---
  datasets.forEach(ds => {{
    h += '<tr><td class="ds">' + ds + '</td>';
    protocols.forEach(p => {{
      const row = filtered.find(
        r => r.dataset === ds && r.protocol === p.key
      );
      subKeys.forEach(sk => {{
        const val   = row ? (row[sk] || 0) : 0;
        const total = row ? (row.total_count || 0) : 0;
        const isCorr  = (sk === 'correct_count');
        const isTot   = (sk === 'total_count');
        const bg = isTot ? '#e9ecef' : cellBg(val, total, isCorr);
        const fw = isTot ? 'font-weight:700;' : '';
        h += '<td style="background:' + bg + ';' + fw + '">' +
             (row ? val : '\u2014') + '</td>';
      }});
    }});
    h += '</tr>';
  }});
  h += '</tbody></table>';

  h = '<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">' +
      '<h3 style="font-size:1.05rem;margin:0;">Checkpoint: ' + checkpoint + '</h3>' +
      '<button onclick="copyTable()" id="btn-copy" style="' +
        'padding:6px 16px;border-radius:6px;border:1px solid #0d6efd;' +
        'background:#0d6efd;color:#fff;font-size:0.82rem;cursor:pointer;' +
        'font-weight:600;">Copy Table</button>' +
      '</div>' + h;
  tableDiv.innerHTML = h;
}}

// -------------------------------------------------------------------
// Copy table to clipboard (preserves formatting for PowerPoint paste)
// -------------------------------------------------------------------
function copyTable() {{
  const table = tableDiv.querySelector('table');
  if (!table) return;
  const btn = document.getElementById('btn-copy');

  // Build a clean HTML table with inline styles so PowerPoint keeps the
  // colours and borders after paste.
  const htmlBlob = new Blob([table.outerHTML], {{ type: 'text/html' }});
  const textBlob = new Blob([table.innerText],  {{ type: 'text/plain' }});

  navigator.clipboard.write([
    new ClipboardItem({{
      'text/html':  htmlBlob,
      'text/plain': textBlob,
    }})
  ]).then(() => {{
    btn.textContent = 'Copied!';
    btn.style.background = '#198754';
    btn.style.borderColor = '#198754';
    setTimeout(() => {{
      btn.textContent = 'Copy Table';
      btn.style.background = '#0d6efd';
      btn.style.borderColor = '#0d6efd';
    }}, 1500);
  }}).catch(() => {{
    // Fallback: select and copy via execCommand
    const range = document.createRange();
    range.selectNode(table);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    document.execCommand('copy');
    window.getSelection().removeAllRanges();
    btn.textContent = 'Copied!';
    setTimeout(() => {{ btn.textContent = 'Copy Table'; }}, 1500);
  }});
}}

// -------------------------------------------------------------------
// Master render dispatcher
// -------------------------------------------------------------------
function render() {{
  const mode = selPrimary.value;
  if (mode === 'comp3') {{
    plotRow.style.display  = 'none';
    tableDiv.style.display = 'block';
    renderComp3();
  }} else {{
    plotRow.style.display  = 'flex';
    tableDiv.style.display = 'none';
    if (mode === 'comp1') renderComp1();
    else                  renderComp2();
  }}
}}

// -------------------------------------------------------------------
// Event wiring and initial load
// -------------------------------------------------------------------
selRun.addEventListener('change', populateSecondary);
selPrimary.addEventListener('change', populateSecondary);
selSecondary.addEventListener('change', render);
populateRuns();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Versioned output naming
# ---------------------------------------------------------------------------

def _max_existing_version(directory, prefix, extension):
    """Return the highest N found in ``<prefix>_v<N>.<ext>`` files, or 0."""
    directory = Path(directory)
    pat = re.compile(
        rf"^{re.escape(prefix)}_v(\d+)\.{re.escape(extension)}$"
    )
    best = 0
    if directory.exists():
        for entry in directory.iterdir():
            m = pat.match(entry.name)
            if m:
                best = max(best, int(m.group(1)))
    return best


def next_shared_version(directory):
    """Determine a single version number shared by both CSV and HTML outputs.

    Scans existing files for both prefixes and picks max(N) + 1 so the
    two output files always carry the same version suffix.
    """
    directory = Path(directory)
    v = max(
        _max_existing_version(directory, CSV_PREFIX, "csv"),
        _max_existing_version(directory, HTML_PREFIX, "html"),
    )
    return v + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_root = RESULTS_ROOT.resolve()
    output_dir = OUTPUT_DIR.resolve()

    rows = collect_rows(results_root)

    if not rows:
        raise SystemExit("No FAILURE_TYPE_SUMMARY.csv files were found.")

    # --- Determine versioned output paths (shared version number) ---
    output_dir.mkdir(parents=True, exist_ok=True)
    version = next_shared_version(output_dir)
    output_csv = output_dir / f"{CSV_PREFIX}_v{version}.csv"
    output_html = output_dir / f"{HTML_PREFIX}_v{version}.html"

    # --- Write CSV ---
    write_csv(rows, output_csv)
    checkpoint_count = len({r["checkpoint"] for r in rows})
    dataset_count = len({r["dataset"] for r in rows})

    print("=" * 78)
    print("HARMONIC-ERROR ANALYSIS")
    print("=" * 78)
    print("Script      :", SCRIPT_VERSION)
    print("Version     :", version)
    print("Checkpoints :", checkpoint_count)
    print("Datasets    :", dataset_count)
    print("Runs        :", len({r["evaluation_run"] for r in rows}))
    print("Setups      :", len(rows))
    print("CSV output  :", output_csv)

    # --- Write HTML report ---
    generate_html_report(rows, output_html)
    print("HTML report :", output_html)
    print("=" * 78)


if __name__ == "__main__":
    main()
