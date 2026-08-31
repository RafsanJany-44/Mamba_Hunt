#!/usr/bin/env python3
"""Generate a complete MAE CSV and interactive Plotly HTML report.

The script scans every evaluation folder below ``results`` and reads the exact
recording-level macro MAE already stored in each protocol's ``summary.json``.
It does not rerun inference, recalculate MAE, or modify evaluation results.
"""

import csv
import json
import re
from pathlib import Path


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
    "aggregation",
    "split_begin",
    "split_end",
    "number_of_recordings",
    "number_of_measurements",
    "mae_bpm",
    "mae_standard_error_bpm",
    "recording_macro_rmse_bpm",
    "accuracy_within_5_bpm_percent",
    "completion_status",
]

RESULTS_ROOT = Path("results")
OUTPUT_DIR = Path("results/mae_analysis")
CSV_PREFIX = "MAE_ALL_SETUPS"
HTML_PREFIX = "MAE_ANALYSIS"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def checkpoint_name(folder_name):
    return KNOWN_CHECKPOINT_NAMES.get(folder_name, folder_name)


def parse_result_path(results_root, summary_path):
    relative = summary_path.relative_to(results_root)
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
    return evaluation_run, checkpoint_folder, dataset_key, protocol


def number(value, default=0.0):
    if value is None:
        return default
    return float(value)


def collect_rows(results_root):
    if not results_root.exists():
        raise SystemExit("Results folder was not found: " + str(results_root))

    rows = []
    seen = set()
    for summary_path in results_root.rglob("summary.json"):
        resolved = summary_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        parsed = parse_result_path(results_root, summary_path)
        if parsed is None:
            continue
        evaluation_run, checkpoint_folder, dataset_key, protocol = parsed

        with summary_path.open("r", encoding="utf-8-sig") as file:
            summary = json.load(file)

        required = "primary_recording_macro_mae_bpm"
        if required not in summary:
            print("Skipped summary without MAE:", summary_path)
            continue

        rows.append({
            "evaluation_run": evaluation_run,
            "checkpoint": checkpoint_name(checkpoint_folder),
            "dataset": DATASET_NAMES.get(dataset_key, dataset_key),
            "protocol": protocol,
            "aggregation": summary.get("aggregation", ""),
            "split_begin": number(summary.get("split_begin")),
            "split_end": number(summary.get("split_end")),
            "number_of_recordings": int(summary.get("number_of_recordings", 0)),
            "number_of_measurements": int(summary.get("number_of_measurements", 0)),
            "mae_bpm": number(summary.get(required)),
            "mae_standard_error_bpm": number(
                summary.get("recording_macro_mae_standard_error_bpm")
            ),
            "recording_macro_rmse_bpm": number(
                summary.get("recording_macro_rmse_bpm")
            ),
            "accuracy_within_5_bpm_percent": number(
                summary.get("accuracy_within_5_bpm_percent")
            ),
            "completion_status": summary.get("completion_status", ""),
        })

    protocol_order = {"Official": 0, "Old": 1, "PRISM": 2}
    rows.sort(key=lambda row: (
        row["evaluation_run"],
        row["checkpoint"],
        row["dataset"],
        protocol_order.get(row["protocol"], 99),
    ))
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def get_plotly_js():
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs()
    except Exception:
        pass

    try:
        from urllib.request import urlopen
        print("Downloading Plotly.js from CDN ...")
        with urlopen(PLOTLY_CDN, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception:
        return None


def html_template():
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RhythmMamba — MAE Analysis</title>
__PLOTLY__
<style>
:root { --bg:#f8f9fa; --card:#fff; --border:#dee2e6; --text:#212529; }
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Segoe UI',system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); padding:24px; }
h1 { font-size:1.6rem; font-weight:700; margin-bottom:4px; }
.subtitle { font-size:.9rem; color:#6c757d; margin-bottom:20px; }
.controls { display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-bottom:20px; background:var(--card); padding:14px 20px; border-radius:10px; border:1px solid var(--border); }
.controls label { font-weight:600; font-size:.85rem; color:#495057; }
.controls select { padding:6px 12px; border-radius:6px; border:1px solid var(--border); font-size:.9rem; background:#fff; cursor:pointer; min-width:220px; }
.plot-row { display:flex; flex-direction:column; gap:24px; margin-bottom:20px; }
.plot-card { width:100%; background:var(--card); border-radius:10px; border:1px solid var(--border); padding:16px; }
#table-container { background:var(--card); border-radius:10px; border:1px solid var(--border); padding:20px; overflow-x:auto; display:none; }
#table-container table { border-collapse:collapse; width:100%; font-size:.85rem; }
#table-container th,#table-container td { border:1px solid var(--border); padding:8px 11px; text-align:center; white-space:nowrap; }
#table-container thead th { background:#343a40; color:#fff; font-weight:600; position:sticky; top:0; }
#table-container thead tr:nth-child(2) th { background:#495057; font-weight:500; }
#table-container tbody td.ds { font-weight:600; background:#e9ecef; text-align:left; }
</style>
</head>
<body>
<h1>RhythmMamba — MAE Analysis</h1>
<p class="subtitle">Recording-level macro MAE comparison across evaluation runs, checkpoints, datasets, and protocols</p>
<div class="controls">
  <label for="sel-run">Run:</label><select id="sel-run"></select>
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
const DATA = __DATA__;
const WINDOW_PROTOCOLS=['Old','PRISM'];
const RECORDING_PROTOCOL='Official';
const COLORS={Official:'#0d6efd',Old:'#e67e22',PRISM:'#e74c3c'};
const LIGHT={Official:'#75a7ef',Old:'#f0b27a',PRISM:'#f1948a'};
function unique(a,f){return [...new Set(a.map(r=>r[f]))];}
const selRun=document.getElementById('sel-run');
const selPrimary=document.getElementById('sel-primary');
const selSecondary=document.getElementById('sel-secondary');
const lblSecondary=document.getElementById('lbl-secondary');
const plotRow=document.getElementById('plot-row');
const tableDiv=document.getElementById('table-container');
function runData(){const v=selRun.value;return v==='__ALL__'?DATA:DATA.filter(r=>r.evaluation_run===v);}
function populateRuns(){selRun.innerHTML='';selRun.add(new Option('All runs','__ALL__'));unique(DATA,'evaluation_run').forEach(v=>selRun.add(new Option(v,v)));populateSecondary();}
function populateSecondary(){const rd=runData();selSecondary.innerHTML='';if(selPrimary.value==='comp1'){lblSecondary.textContent='Dataset:';unique(rd,'dataset').forEach(v=>selSecondary.add(new Option(v,v)));}else{lblSecondary.textContent='Checkpoint:';unique(rd,'checkpoint').forEach(v=>selSecondary.add(new Option(v,v)));}render();}
function buildBar(divId,filtered,groupField,protocols,title){
  const groups=unique(filtered,groupField);const traces=[];
  protocols.forEach(proto=>{
    const pr=filtered.filter(r=>r.protocol===proto);
    const mae=[],se=[],n=[];
    groups.forEach(g=>{const r=pr.find(x=>x[groupField]===g);mae.push(r?r.mae_bpm:0);se.push(r?r.mae_standard_error_bpm:0);n.push(r?r.number_of_recordings:0);});
    traces.push({name:proto,x:groups,y:mae,error_y:{type:'data',array:se,visible:true,color:COLORS[proto],thickness:1.2},customdata:n,type:'bar',text:mae.map(v=>v?v.toFixed(3):''),textposition:'outside',cliponaxis:false,marker:{color:proto==='PRISM'?LIGHT[proto]:COLORS[proto],line:{color:COLORS[proto],width:1}},hovertemplate:'<b>'+proto+'</b><br>%{x}<br>MAE: %{y:.3f} BPM<br>Recordings: %{customdata}<extra></extra>'});
  });
  Plotly.newPlot(divId,traces,{title:{text:title,font:{size:15}},barmode:'group',xaxis:{title:{text:groupField==='checkpoint'?'Checkpoint':'Dataset'},tickangle:-25,automargin:true},yaxis:{title:{text:'Recording-level macro MAE (BPM)'},rangemode:'tozero'},legend:{orientation:'h',y:-.3,x:.5,xanchor:'center'},margin:{t:55,b:130,l:70,r:20},plot_bgcolor:'#fafafa',paper_bgcolor:'#fff'}, {responsive:true});
}
function renderComp1(){const ds=selSecondary.value;const f=runData().filter(r=>r.dataset===ds);buildBar('plot-left',f.filter(r=>WINDOW_PROTOCOLS.includes(r.protocol)),'checkpoint',WINDOW_PROTOCOLS,'Old & PRISM — '+ds);buildBar('plot-right',f.filter(r=>r.protocol===RECORDING_PROTOCOL),'checkpoint',[RECORDING_PROTOCOL],'Official — '+ds);}
function renderComp2(){const cp=selSecondary.value;const f=runData().filter(r=>r.checkpoint===cp);buildBar('plot-left',f.filter(r=>WINDOW_PROTOCOLS.includes(r.protocol)),'dataset',WINDOW_PROTOCOLS,'Old & PRISM — '+cp);buildBar('plot-right',f.filter(r=>r.protocol===RECORDING_PROTOCOL),'dataset',[RECORDING_PROTOCOL],'Official — '+cp);}
function cellBg(value,max){if(value===null)return'#f8f9fa';if(max<=0)return'#d4edda';const q=Math.min(value/max,1);const r=Math.round(210+45*q),g=Math.round(238-125*q),b=Math.round(218-105*q);return'rgb('+r+','+g+','+b+')';}
function renderComp3(){
  const cp=selSecondary.value,f=runData().filter(r=>r.checkpoint===cp),datasets=unique(f,'dataset'),protocols=['Official','Old','PRISM'];
  const max=Math.max(...f.map(r=>r.mae_bpm),0);let h='<table><thead><tr><th rowspan="2">Eval Dataset</th>';protocols.forEach(p=>h+='<th colspan="2">'+p+'</th>');h+='</tr><tr>';protocols.forEach(()=>h+='<th>MAE ± SE (BPM)</th><th>N</th>');h+='</tr></thead><tbody>';
  datasets.forEach(ds=>{h+='<tr><td class="ds">'+ds+'</td>';protocols.forEach(p=>{const r=f.find(x=>x.dataset===ds&&x.protocol===p);if(r){h+='<td style="background:'+cellBg(r.mae_bpm,max)+';font-weight:600">'+r.mae_bpm.toFixed(3)+' ± '+r.mae_standard_error_bpm.toFixed(3)+'</td><td>'+r.number_of_recordings+'</td>';}else h+='<td>—</td><td>—</td>';});h+='</tr>';});h+='</tbody></table>';
  h='<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px"><h3 style="font-size:1.05rem;margin:0">Checkpoint: '+cp+'</h3><button onclick="copyTable()" id="btn-copy" style="padding:6px 16px;border-radius:6px;border:1px solid #0d6efd;background:#0d6efd;color:#fff;font-size:.82rem;cursor:pointer;font-weight:600">Copy Table</button></div>'+h;tableDiv.innerHTML=h;
}
function copyTable(){const table=tableDiv.querySelector('table');if(!table)return;const btn=document.getElementById('btn-copy');const htmlBlob=new Blob([table.outerHTML],{type:'text/html'}),textBlob=new Blob([table.innerText],{type:'text/plain'});navigator.clipboard.write([new ClipboardItem({'text/html':htmlBlob,'text/plain':textBlob})]).then(()=>{btn.textContent='Copied!';btn.style.background='#198754';setTimeout(()=>{btn.textContent='Copy Table';btn.style.background='#0d6efd';},1500);}).catch(()=>{const range=document.createRange();range.selectNode(table);window.getSelection().removeAllRanges();window.getSelection().addRange(range);document.execCommand('copy');window.getSelection().removeAllRanges();btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy Table',1500);});}
function render(){if(selPrimary.value==='comp3'){plotRow.style.display='none';tableDiv.style.display='block';renderComp3();}else{plotRow.style.display='flex';tableDiv.style.display='none';if(selPrimary.value==='comp1')renderComp1();else renderComp2();}}
selRun.addEventListener('change',populateSecondary);selPrimary.addEventListener('change',populateSecondary);selSecondary.addEventListener('change',render);populateRuns();
</script>
</body>
</html>'''


def generate_html(rows, output_path):
    plotly_js = get_plotly_js()
    if plotly_js is None:
        print("WARNING: Plotly.js could not be embedded; using the CDN link.")
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'
    else:
        plotly_tag = "<script>\n" + plotly_js + "\n</script>"
    html = html_template().replace("__PLOTLY__", plotly_tag).replace(
        "__DATA__", json.dumps(rows, separators=(",", ":"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def max_version(directory, prefix, extension):
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.{extension}$")
    best = 0
    if directory.exists():
        for path in directory.iterdir():
            match = pattern.match(path.name)
            if match:
                best = max(best, int(match.group(1)))
    return best


def next_shared_version(directory):
    return max(
        max_version(directory, CSV_PREFIX, "csv"),
        max_version(directory, HTML_PREFIX, "html"),
    ) + 1


def main():
    results_root = RESULTS_ROOT.resolve()
    output_dir = OUTPUT_DIR.resolve()
    rows = collect_rows(results_root)
    if not rows:
        raise SystemExit("No evaluation summary.json files containing MAE were found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    version = next_shared_version(output_dir)
    output_csv = output_dir / f"{CSV_PREFIX}_v{version}.csv"
    output_html = output_dir / f"{HTML_PREFIX}_v{version}.html"
    write_csv(rows, output_csv)
    generate_html(rows, output_html)

    print("=" * 78)
    print("MAE ANALYSIS GENERATED")
    print("=" * 78)
    print("Version     :", version)
    print("Checkpoints :", len({row['checkpoint'] for row in rows}))
    print("Datasets    :", len({row['dataset'] for row in rows}))
    print("Runs        :", len({row['evaluation_run'] for row in rows}))
    print("Setups      :", len(rows))
    print("CSV output  :", output_csv)
    print("HTML output :", output_html)
    print("=" * 78)


if __name__ == "__main__":
    main()
