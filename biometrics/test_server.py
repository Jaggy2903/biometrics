import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import csv
import json
import io
import datetime
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from flask import Flask, jsonify, render_template_string, request, send_from_directory, Response
from deepface import DeepFace

app = Flask(__name__)

DATA_DIR = os.path.join("data", "s01")
VALID_EXT = (".jpg", ".jpeg", ".png")
RESULTS_FILE = "results.csv"

# In-memory session results
session_results = []

HTML = r"""
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <title>DeepFace Analyse</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300..700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0f0f0f;
      --surface: #171717;
      --surface2: #1f1f1f;
      --border: rgba(255,255,255,0.08);
      --text: #e8e8e6;
      --muted: #888;
      --faint: #444;
      --primary: #4f98a3;
      --primary-hover: #3d8491;
      --success: #6daa45;
      --error: #d16363;
      --warning: #e8af34;
      --radius: 10px;
      --radius-sm: 6px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { -webkit-font-smoothing: antialiased; scroll-behavior: smooth; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); font-size: 15px; line-height: 1.6; min-height: 100dvh; }

    header {
      position: sticky; top: 0; z-index: 100;
      background: rgba(15,15,15,0.85); backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex; align-items: center; gap: 10px;
    }
    header svg { color: var(--primary); flex-shrink: 0; }
    header h1 { font-size: 17px; font-weight: 600; }
    header span { color: var(--muted); font-size: 13px; margin-left: 4px; }

    main { max-width: 1100px; margin: 0 auto; padding: 28px 20px; }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 700px) { .grid-2 { grid-template-columns: 1fr; } }

    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 18px;
    }
    .card h2 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 14px; }

    label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 5px; }
    select {
      width: 100%; padding: 9px 12px; background: var(--surface2); border: 1px solid var(--border);
      border-radius: var(--radius-sm); color: var(--text); font-size: 14px; cursor: pointer;
      transition: border-color 180ms;
    }
    select:focus { outline: none; border-color: var(--primary); }

    .preview-img {
      width: 100%; aspect-ratio: 4/3; object-fit: contain;
      background: var(--surface2); border-radius: var(--radius-sm);
      margin-top: 10px; border: 1px solid var(--border);
    }

    .btn {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 10px 20px; border-radius: var(--radius-sm);
      font-size: 14px; font-weight: 500; cursor: pointer;
      border: none; transition: background 180ms;
    }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-ghost {
      background: transparent; color: var(--muted);
      border: 1px solid var(--border);
    }
    .btn-ghost:hover { background: var(--surface2); color: var(--text); }
    .btn-sm { padding: 7px 14px; font-size: 13px; }

    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }

    #result-box {
      margin-top: 16px; padding: 16px;
      background: var(--surface2); border-radius: var(--radius-sm);
      border: 1px solid var(--border); min-height: 80px;
      display: flex; flex-direction: column; gap: 8px;
    }
    .verdict {
      font-size: 22px; font-weight: 700; display: flex; align-items: center; gap: 8px;
    }
    .verdict.match { color: var(--success); }
    .verdict.reject { color: var(--error); }

    .stats-row {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px;
      margin-top: 4px;
    }
    .stat {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius-sm); padding: 10px 12px;
    }
    .stat-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    .stat-value { font-size: 18px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }

    /* distance bar */
    .dist-bar-wrap { margin-top: 8px; }
    .dist-bar-bg {
      height: 8px; background: var(--faint); border-radius: 99px; overflow: hidden;
    }
    .dist-bar-fill {
      height: 100%; border-radius: 99px;
      transition: width .5s cubic-bezier(0.16,1,0.3,1);
      background: var(--primary);
    }
    .dist-bar-labels { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 3px; }

    /* threshold line marker */
    .dist-bar-outer { position: relative; }
    .threshold-marker {
      position: absolute; top: -4px; bottom: -4px;
      width: 2px; background: var(--error);
      border-radius: 2px;
    }
    .threshold-label {
      position: absolute; top: -18px;
      font-size: 10px; color: var(--error);
      transform: translateX(-50%);
      white-space: nowrap;
    }

    /* history table */
    .table-wrap { overflow-x: auto; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 8px 10px; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    td { padding: 8px 10px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
    tr:last-child td { border-bottom: none; }
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 99px;
      font-size: 11px; font-weight: 600;
    }
    .badge-match { background: rgba(109,170,69,.18); color: var(--success); }
    .badge-reject { background: rgba(209,99,99,.18); color: var(--error); }

    .chart-wrap { position: relative; height: 220px; margin-top: 12px; }

    .empty-state { text-align: center; padding: 32px; color: var(--faint); font-size: 13px; }

    .spinner {
      display: inline-block; width: 16px; height: 16px;
      border: 2px solid var(--border); border-top-color: var(--primary);
      border-radius: 50%; animation: spin .7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<header>
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
    <circle cx="12" cy="8" r="4"/><path d="M6 20v-2a6 6 0 0 1 12 0v2"/>
    <path d="M16 3.5a4 4 0 0 1 0 7M20 20v-2a6 6 0 0 0-4-5.65"/>
  </svg>
  <h1>DeepFace Analyse <span>VGG-Face · cosine · data/s01</span></h1>
</header>

<main>
  <!-- Foto selectie -->
  <div class="grid-2" style="margin-bottom:16px">
    <div class="card">
      <h2>Reference foto</h2>
      <label for="img1">Kies bestand</label>
      <select id="img1" onchange="updatePreview('img1','prev1')"></select>
      <img id="prev1" class="preview-img" src="" alt="reference preview">
    </div>
    <div class="card">
      <h2>Test foto</h2>
      <label for="img2">Kies bestand</label>
      <select id="img2" onchange="updatePreview('img2','prev2')"></select>
      <img id="prev2" class="preview-img" src="" alt="test preview">
    </div>
  </div>

  <!-- Actieknoppen -->
  <div class="actions">
    <button class="btn btn-primary" onclick="runVerify()" id="btn-verify">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>
      Vergelijk
    </button>
    <button class="btn btn-ghost btn-sm" onclick="downloadCSV()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Export CSV
    </button>
    <button class="btn btn-ghost btn-sm" onclick="clearHistory()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/></svg>
      Wis sessie
    </button>
  </div>

  <!-- Resultaat -->
  <div id="result-box" style="margin-top:16px">
    <span style="color:var(--faint);font-size:13px">Resultaat verschijnt hier na vergelijking.</span>
  </div>

  <!-- Grafiek + Tabel -->
  <div class="grid-2" style="margin-top:20px">
    <div class="card">
      <h2>Cosine distance per vergelijking</h2>
      <div class="chart-wrap">
        <canvas id="chart"></canvas>
      </div>
    </div>
    <div class="card">
      <h2>Sessie geschiedenis</h2>
      <div class="table-wrap" id="history-wrap">
        <div class="empty-state">Nog geen vergelijkingen gedaan.</div>
      </div>
    </div>
  </div>
</main>

<script>
let sessionData = [];
let chartInstance = null;

async function loadFiles() {
  const res = await fetch('/files');
  const files = await res.json();
  const s1 = document.getElementById('img1');
  const s2 = document.getElementById('img2');
  s1.innerHTML = s2.innerHTML = '';
  files.forEach(f => {
    s1.appendChild(new Option(f, f));
    s2.appendChild(new Option(f, f));
  });
  if (files.length > 0) {
    s1.value = files.includes('reference.jpg') ? 'reference.jpg' : files[0];
    s2.value = files.includes('90cm.jpg') ? '90cm.jpg' : files[Math.min(1, files.length-1)];
    updatePreview('img1','prev1');
    updatePreview('img2','prev2');
  }
}

function updatePreview(selectId, imgId) {
  const v = document.getElementById(selectId).value;
  document.getElementById(imgId).src = v ? '/image/' + v : '';
}

async function runVerify() {
  const img1 = document.getElementById('img1').value;
  const img2 = document.getElementById('img2').value;
  const box = document.getElementById('result-box');
  const btn = document.getElementById('btn-verify');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Bezig...';
  box.innerHTML = '<span style="color:var(--muted);font-size:13px">⏳ DeepFace analyseren...</span>';

  const res = await fetch('/verify', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({img1, img2})
  });
  const data = await res.json();

  btn.disabled = false;
  btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg> Vergelijk`;

  if (data.error) {
    box.innerHTML = `<span style="color:var(--error)">Fout: ${data.error}</span>`;
    return;
  }

  const ts = new Date().toLocaleTimeString('nl-NL');
  const pct = Math.min(100, (data.distance / 1.0) * 100);
  const thresholdPct = data.threshold * 100;

  box.innerHTML = `
    <div class="verdict ${data.verified ? 'match' : 'reject'}">
      ${data.verified
        ? '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> MATCH'
        : '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg> REJECT'}
    </div>
    <div class="stats-row">
      <div class="stat"><div class="stat-label">Cosine distance</div><div class="stat-value" style="color:${data.verified?'var(--success)':'var(--error)'}">${data.distance.toFixed(4)}</div></div>
      <div class="stat"><div class="stat-label">Threshold</div><div class="stat-value">${data.threshold.toFixed(2)}</div></div>
      <div class="stat"><div class="stat-label">Marge</div><div class="stat-value" style="color:${data.margin>=0?'var(--success)':'var(--error)'}">${data.margin > 0 ? '+' : ''}${data.margin.toFixed(4)}</div></div>
      <div class="stat"><div class="stat-label">Conf. score</div><div class="stat-value">${data.confidence}%</div></div>
    </div>
    <div class="dist-bar-wrap">
      <div class="dist-bar-outer">
        <div class="dist-bar-bg">
          <div class="dist-bar-fill" style="width:${pct}%;background:${data.verified?'var(--success)':'var(--error)'}"></div>
        </div>
        <div class="threshold-marker" style="left:${thresholdPct}%">
          <div class="threshold-label">drempel ${data.threshold}</div>
        </div>
      </div>
      <div class="dist-bar-labels"><span>0.0</span><span>0.5</span><span>1.0</span></div>
    </div>
    <div style="font-size:12px;color:var(--muted);margin-top:4px">${img1} vs. ${img2} — ${ts}</div>
  `;

  // Save to session
  sessionData.push({
    timestamp: ts, img1, img2,
    distance: data.distance,
    threshold: data.threshold,
    margin: data.margin,
    confidence: data.confidence,
    verified: data.verified
  });

  updateChart();
  updateTable();
}

function updateChart() {
  const labels = sessionData.map((d,i) => `#${i+1}`);
  const distances = sessionData.map(d => d.distance);
  const colors = sessionData.map(d => d.verified ? 'rgba(109,170,69,0.85)' : 'rgba(209,99,99,0.85)');
  const threshold = sessionData.length > 0 ? sessionData[0].threshold : 0.40;

  if (chartInstance) chartInstance.destroy();

  chartInstance = new Chart(document.getElementById('chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Cosine distance',
          data: distances,
          backgroundColor: colors,
          borderRadius: 4,
          borderSkipped: false,
        },
        {
          label: `Threshold (${threshold})`,
          data: Array(distances.length).fill(threshold),
          type: 'line',
          borderColor: 'rgba(232,175,52,0.8)',
          borderWidth: 2,
          borderDash: [5,4],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#888', font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.datasetIndex === 0) {
                const d = sessionData[ctx.dataIndex];
                return [`Distance: ${d.distance.toFixed(4)}`, `${d.verified ? 'MATCH' : 'REJECT'}`, `${d.img1} vs ${d.img2}`];
              }
              return `Threshold: ${threshold}`;
            }
          }
        }
      },
      scales: {
        x: { ticks: { color:'#666' }, grid: { color:'rgba(255,255,255,0.05)' } },
        y: { min: 0, max: 1, ticks: { color:'#666' }, grid: { color:'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

function updateTable() {
  const wrap = document.getElementById('history-wrap');
  if (sessionData.length === 0) {
    wrap.innerHTML = '<div class="empty-state">Nog geen vergelijkingen gedaan.</div>';
    return;
  }
  const rows = [...sessionData].reverse().map((d, i) => {
    const idx = sessionData.length - i;
    return `<tr>
      <td style="color:var(--muted)">#${idx}</td>
      <td>${d.img1}</td>
      <td>${d.img2}</td>
      <td><b>${d.distance.toFixed(4)}</b></td>
      <td>${d.confidence}%</td>
      <td><span class="badge ${d.verified ? 'badge-match' : 'badge-reject'}">${d.verified ? 'MATCH' : 'REJECT'}</span></td>
      <td style="color:var(--muted)">${d.timestamp}</td>
    </tr>`;
  }).join('');

  wrap.innerHTML = `<table>
    <thead><tr>
      <th>#</th><th>Ref</th><th>Test</th><th>Distance</th><th>Conf.</th><th>Uitslag</th><th>Tijd</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function downloadCSV() {
  if (sessionData.length === 0) { alert('Geen data om te exporteren.'); return; }
  const headers = ['#','timestamp','img1','img2','cosine_distance','threshold','margin','confidence_pct','verified'];
  const rows = sessionData.map((d,i) =>
    [i+1, d.timestamp, d.img1, d.img2, d.distance, d.threshold, d.margin, d.confidence, d.verified].join(',')
  );
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `deepface_results_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.csv`;
  a.click();
}

function clearHistory() {
  if (!confirm('Weet je zeker dat je de sessiegeschiedenis wilt wissen?')) return;
  sessionData = [];
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  document.getElementById('chart').getContext('2d').clearRect(0,0,9999,9999);
  updateTable();
  document.getElementById('result-box').innerHTML = '<span style="color:var(--faint);font-size:13px">Resultaat verschijnt hier na vergelijking.</span>';
}

loadFiles();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/files")
def list_files():
    if not os.path.exists(DATA_DIR):
        return jsonify([])
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(VALID_EXT)])
    return jsonify(files)

@app.route("/image/<path:filename>")
def serve_image(filename):
    return send_from_directory(DATA_DIR, filename)

@app.route("/verify", methods=["POST"])
def verify():
    try:
        data = request.get_json()
        img1 = os.path.join(DATA_DIR, data["img1"])
        img2 = os.path.join(DATA_DIR, data["img2"])

        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=True,
            align=True,
            silent=True,
        )

        dist = round(result["distance"], 6)
        thr  = result["threshold"]
        margin = round(thr - dist, 6)
        # confidence: hoe ver van threshold, als % van threshold range
        confidence = max(0, min(100, round((1 - dist / thr) * 100 if result["verified"] else (1 - (dist - thr) / thr) * 50, 1)))

        # Append to CSV
        write_header = not os.path.exists(RESULTS_FILE)
        with open(RESULTS_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","img1","img2","cosine_distance","threshold","margin","confidence_pct","verified"])
            w.writerow([
                datetime.datetime.now().isoformat(),
                data["img1"], data["img2"],
                dist, thr, margin, confidence, result["verified"]
            ])

        return jsonify({
            "verified": result["verified"],
            "distance": dist,
            "threshold": thr,
            "margin": margin,
            "confidence": confidence,
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
