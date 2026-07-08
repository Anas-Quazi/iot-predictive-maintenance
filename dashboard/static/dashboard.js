// ---------- Palette (multi-color, not single-hue dominant) ----------
const NEON = {
  cyan: '#00e5ff', green: '#39ff8a', pink: '#ff2e88', amber: '#ffb300',
  purple: '#b26bff', blue: '#4d7bff', red: '#ff3b5c',
};
const tierColor = { Low: NEON.green, Medium: NEON.amber, High: '#ff8a3d', Critical: NEON.red };
Chart.defaults.color = '#8891a8';
Chart.defaults.borderColor = '#1c2130';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ---------- Tab switching ----------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'insights') loadInsights();
    if (btn.dataset.tab === 'replay') loadSessions();
  });
});

let charts = {};
function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

// ---------- Fleet polling ----------
async function refreshFleet() {
  try {
    const res = await fetch('/api/fleet_status');
    const data = await res.json();
    renderKpis(data.kpis);
    renderFleetTable(data.machines);
    renderAlerts(data.alerts);
    populateMachineSelect(data.machines);
  } catch (e) { console.error('fleet refresh failed', e); }
}

function renderKpis(kpis) {
  const bar = document.getElementById('kpi-bar');
  bar.innerHTML = `
    <div class="kpi">Machines: <b>${kpis.total_machines}</b></div>
    <div class="kpi" style="color:${tierColor.Low}">Low: ${kpis.tier_counts.Low}</div>
    <div class="kpi" style="color:${tierColor.Medium}">Medium: ${kpis.tier_counts.Medium}</div>
    <div class="kpi" style="color:${tierColor.High}">High: ${kpis.tier_counts.High}</div>
    <div class="kpi" style="color:${tierColor.Critical}">Critical: ${kpis.tier_counts.Critical}</div>
  `;
}

function renderFleetTable(machines) {
  const tbody = document.querySelector('#fleet-table tbody');
  tbody.innerHTML = machines.map(m => {
    const deltaClass = m.delta > 0.001 ? 'delta-up' : (m.delta < -0.001 ? 'delta-down' : '');
    const arrow = m.delta > 0.001 ? '▲' : (m.delta < -0.001 ? '▼' : '—');
    return `<tr class="tier-${m.risk_tier}">
      <td>${m.machine_id}</td>
      <td>${(m.probability * 100).toFixed(2)}%</td>
      <td><span class="badge ${m.risk_tier}">${m.risk_tier}</span></td>
      <td class="${deltaClass}">${arrow} ${(m.delta*100).toFixed(2)}%</td>
      <td>${m.primary_reason}</td>
      <td>${m.tool_wear?.toFixed(1) ?? '-'}</td>
      <td>${m.process_temp?.toFixed(1) ?? '-'}</td>
    </tr>`;
  }).join('');
}

function renderAlerts(alerts) {
  const list = document.getElementById('alerts-list');
  if (!alerts.length) { list.innerHTML = '<li>No alerts yet.</li>'; return; }
  list.innerHTML = alerts.map(a =>
    `<li><span class="badge ${a.tier === 'MAINTENANCE_RESET' ? 'Low' : a.tier}">${a.tier}</span>
     ${a.machine_id} — ${(a.probability*100).toFixed(2)}% — ${new Date(a.timestamp).toLocaleTimeString()}</li>`
  ).join('');
}

let selectedMachine = null;
function populateMachineSelect(machines) {
  const sel = document.getElementById('machine-select');
  const ids = machines.map(m => m.machine_id);
  if (sel.dataset.filled === ids.join(',')) return;
  sel.dataset.filled = ids.join(',');
  sel.innerHTML = ids.map(id => `<option value="${id}">${id}</option>`).join('');
  if (!selectedMachine && ids.length) selectedMachine = ids[0];
  sel.value = selectedMachine;
}
document.getElementById('machine-select').addEventListener('change', e => {
  selectedMachine = e.target.value;
  loadMachineDetail();
});

async function loadMachineDetail() {
  if (!selectedMachine) return;
  const res = await fetch(`/api/machine/${selectedMachine}`);
  if (!res.ok) return;
  const data = await res.json();
  renderGauge(data.result.probability, data.result.risk_tier);
  renderShap('shap-chart', data.result.top_drivers);
  renderTrend(data.raw_history);
  renderRiskHistory(data.risk_history);
  document.getElementById('action-text').textContent =
    `${data.result.primary_reason}  →  ${data.result.recommended_action}`;
}

function renderGauge(prob, tier) {
  destroyChart('gauge');
  const ctx = document.getElementById('gauge-chart');
  charts['gauge'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Failure risk', ''],
      datasets: [{ data: [prob * 100, 100 - prob * 100], backgroundColor: [tierColor[tier], '#12141c'], borderWidth: 0 }]
    },
    options: {
      circumference: 270, rotation: 225, cutout: '75%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    },
    plugins: [{
      id: 'centerText',
      afterDraw(chart) {
        const { ctx, chartArea } = chart;
        ctx.save();
        ctx.font = 'bold 28px sans-serif';
        ctx.fillStyle = tierColor[tier];
        ctx.textAlign = 'center';
        ctx.fillText((prob * 100).toFixed(1) + '%', (chartArea.left + chartArea.right) / 2, (chartArea.top + chartArea.bottom) / 2 + 10);
        ctx.restore();
      }
    }]
  });
}

function renderShap(canvasId, drivers, chartKey) {
  const key = chartKey || canvasId;
  destroyChart(key);
  const ctx = document.getElementById(canvasId);
  charts[key] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: drivers.map(d => d.feature),
      datasets: [{
        data: drivers.map(d => d.impact),
        backgroundColor: drivers.map(d => d.impact >= 0 ? NEON.red : NEON.green),
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'SHAP impact on failure probability' } } }
    }
  });
}

function renderTrend(history) {
  destroyChart('trend');
  const ctx = document.getElementById('trend-chart');
  const labels = history.map((_, i) => i);
  charts['trend'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Tool Wear (min)', data: history.map(h => h.tool_wear), borderColor: NEON.amber, yAxisID: 'y', tension: .3 },
        { label: 'Torque (Nm)', data: history.map(h => h.torque), borderColor: NEON.cyan, yAxisID: 'y1', tension: .3 },
        { label: 'Process Temp (K)', data: history.map(h => h.process_temp), borderColor: NEON.pink, yAxisID: 'y1', tension: .3 },
      ]
    },
    options: {
      scales: { y: { position: 'left' }, y1: { position: 'right', grid: { drawOnChartArea: false } } }
    }
  });
}

function renderRiskHistory(history) {
  destroyChart('riskhist');
  const ctx = document.getElementById('risk-history-chart');
  charts['riskhist'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: history.map(h => new Date(h.t).toLocaleTimeString()),
      datasets: [{ label: 'Failure probability', data: history.map(h => h.p * 100), borderColor: NEON.red, fill: true, backgroundColor: 'rgba(255,59,92,.12)', tension: .3 }]
    },
    options: { scales: { y: { min: 0, max: 100, title: { display: true, text: '%' } } } }
  });
}

// ---------- Manual / What-if ----------
document.getElementById('manual-form').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  if (!payload.borrow_history_timestamp) delete payload.borrow_history_timestamp;
  const res = await fetch('/api/manual_predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await res.json();
  const box = document.getElementById('manual-result');
  if (data.error) { box.innerHTML = `<p>Error: ${data.error}</p>`; return; }
  box.innerHTML = `
    <div class="card">
      <h3>Result</h3>
      <p>Probability: <b style="color:${tierColor[data.result.risk_tier]}">${(data.result.probability*100).toFixed(2)}%</b>
         — Tier: <span class="badge ${data.result.risk_tier}">${data.result.risk_tier}</span></p>
      <p>${data.result.primary_reason}</p>
      <p><b>Recommended action:</b> ${data.result.recommended_action}</p>
      ${data.cold_start_warning ? '<p class="warning-text">⚠ No history borrowed — rolling/lag features defaulted to this single reading. Result is a cold-start estimate.</p>' : ''}
    </div>
  `;
  document.getElementById('manual-charts').style.display = 'grid';
  renderManualRangeChart(data.entered_values, data.normal_ranges);
  renderShap('manual-shap-chart', data.result.top_drivers, 'manualshap');
});

function renderManualRangeChart(entered, ranges) {
  destroyChart('manualrange');
  const ctx = document.getElementById('manual-range-chart');
  const fields = ['air_temp', 'process_temp', 'rpm', 'torque', 'tool_wear'];
  const labels = ['Air Temp', 'Process Temp', 'RPM', 'Torque', 'Tool Wear'];
  charts['manualrange'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Normal range (5th-95th pct)',
          data: fields.map(f => [ranges[f].p5, ranges[f].p95]),
          backgroundColor: 'rgba(77,123,255,.25)',
          borderColor: NEON.blue, borderWidth: 1,
        },
        {
          label: 'Entered value',
          data: fields.map(f => entered[f]),
          type: 'scatter',
          backgroundColor: NEON.pink,
          pointRadius: 7,
          pointStyle: 'rectRot',
        }
      ]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { position: 'bottom' } },
      scales: { x: { title: { display: true, text: 'Value (each row own scale approx.)' } } }
    }
  });
}

// ---------- Timestamp lookup ----------
document.getElementById('lookup-form').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  const res = await fetch('/api/lookup_timestamp', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await res.json();
  const box = document.getElementById('lookup-result');
  if (data.error) { box.innerHTML = `<p>Error: ${data.error}</p>`; return; }
  box.innerHTML = `
    <div class="card">
      <h3>Prediction vs. Actual — ${data.timestamp}</h3>
      <p>Predicted probability: <b style="color:${tierColor[data.result.risk_tier]}">${(data.result.probability*100).toFixed(2)}%</b>
         (${data.result.risk_tier}) &nbsp;|&nbsp; Actual failure recorded: <b>${data.actual_failure_label ? 'YES' : 'no'}</b></p>
      <p>Actual failure types at this timestamp: ${Object.entries(data.actual_failure_types).filter(([,v])=>v).map(([k])=>k).join(', ') || 'none'}</p>
      <p>${data.result.primary_reason} → ${data.result.recommended_action}</p>
      <p><b>Top drivers:</b> ${data.result.top_drivers.map(d => `${d.feature} (${d.impact>0?'+':''}${d.impact.toFixed(3)})`).join(', ')}</p>
    </div>
  `;
});

// ---------- Timestamp range ----------
document.getElementById('range-form').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  const res = await fetch('/api/timestamp_range', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await res.json();
  const box = document.getElementById('range-result');
  if (data.error) { box.innerHTML = `<p>Error: ${data.error}</p>`; return; }

  box.innerHTML = `
    <div class="kpi-grid">
      <div class="stat-card"><div class="label">Rows analyzed</div><div class="value">${data.n_rows}</div></div>
      <div class="stat-card"><div class="label">Avg probability</div><div class="value" style="color:${NEON.cyan}">${(data.avg_probability*100).toFixed(2)}%</div></div>
      <div class="stat-card"><div class="label">Max probability</div><div class="value" style="color:${NEON.red}">${(data.max_probability*100).toFixed(2)}%</div></div>
      <div class="stat-card"><div class="label">Actual failures in range</div><div class="value" style="color:${NEON.amber}">${data.actual_failures_in_range}</div></div>
      <div class="stat-card"><div class="label">Critical readings</div><div class="value" style="color:${NEON.red}">${data.tier_counts.Critical}</div></div>
    </div>
    <div class="detail-grid">
      <div class="card wide">
        <h3>Risk Trajectory</h3>
        <canvas id="range-trajectory-chart"></canvas>
      </div>
      <div class="card">
        <h3>Top Drivers Across This Range</h3>
        <canvas id="range-shap-chart"></canvas>
      </div>
      <div class="card">
        <h3>Environment Averages (this window)</h3>
        <table><tbody>
          ${Object.entries(data.environment_avg).map(([k,v]) => `<tr><td>${k}</td><td>${v.toFixed(2)}</td></tr>`).join('')}
        </tbody></table>
      </div>
      <div class="card wide">
        <h3>Recommendation</h3>
        <p>${data.recommendation}</p>
        <p style="color:var(--muted);font-size:12.5px">Failure-type occurrences in this window (descriptive):
          ${Object.entries(data.actual_failure_type_counts).map(([k,v])=>`${k}: ${v}`).join(' · ')}</p>
      </div>
    </div>
  `;

  destroyChart('rangetraj');
  charts['rangetraj'] = new Chart(document.getElementById('range-trajectory-chart'), {
    type: 'line',
    data: {
      labels: data.trajectory.map(t => t.timestamp),
      datasets: [
        { label: 'Predicted probability (%)', data: data.trajectory.map(t => t.probability*100), borderColor: NEON.cyan, tension: .2, pointRadius: 0 },
        { label: 'Actual failure', data: data.trajectory.map(t => t.actual_failure*100), borderColor: NEON.red, borderDash: [3,3], pointRadius: 0 },
      ]
    },
    options: { scales: { x: { ticks: { maxTicksLimit: 8 } } } }
  });

  destroyChart('rangeshap');
  charts['rangeshap'] = new Chart(document.getElementById('range-shap-chart'), {
    type: 'bar',
    data: {
      labels: data.top_drivers_overall.map(d => d.feature),
      datasets: [{ data: data.top_drivers_overall.map(d => d.avg_abs_impact), backgroundColor: NEON.purple }]
    },
    options: { indexAxis: 'y', plugins: { legend: { display: false } } }
  });
});

// ---------- Session replay ----------
async function loadSessions() {
  const sel = document.getElementById('session-select');
  if (sel.dataset.loaded) return;
  const res = await fetch('/api/sessions');
  const sessions = await res.json();
  sel.innerHTML = sessions.map(s =>
    `<option value="${s.session_index}">#${s.session_index} — ${s.start_timestamp} → ${s.end_timestamp} (${s.n_rows} rows)${s.had_failure ? ' ⚠ had failure' : ''}</option>`
  ).join('');
  sel.dataset.loaded = '1';
}
document.getElementById('replay-btn').addEventListener('click', async () => {
  const idx = document.getElementById('session-select').value;
  const res = await fetch('/api/machine_range', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_index: idx }) });
  const data = await res.json();
  destroyChart('replay');
  const ctx = document.getElementById('replay-chart');
  charts['replay'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.trajectory.map(t => t.timestamp),
      datasets: [
        { label: 'Predicted failure probability (%)', data: data.trajectory.map(t => t.probability * 100), borderColor: NEON.cyan, tension: .2, pointRadius: 0 },
        { label: 'Actual failure (0/100)', data: data.trajectory.map(t => t.actual_failure * 100), borderColor: NEON.red, borderDash: [4,4], pointRadius: 0 },
      ]
    },
    options: { scales: { x: { ticks: { maxTicksLimit: 10 } } } }
  });
});

// ---------- Global insights ----------
async function loadInsights() {
  const res = await fetch('/api/global_insights');
  const data = await res.json();

  document.getElementById('insights-kpis').innerHTML = `
    <div class="stat-card"><div class="label">Accuracy</div><div class="value" style="color:${NEON.green}">${(data.model_metrics.accuracy*100).toFixed(2)}%</div></div>
    <div class="stat-card"><div class="label">F1 Score</div><div class="value" style="color:${NEON.cyan}">${data.model_metrics.f1.toFixed(3)}</div></div>
    <div class="stat-card"><div class="label">Precision</div><div class="value" style="color:${NEON.purple}">${data.model_metrics.precision.toFixed(3)}</div></div>
    <div class="stat-card"><div class="label">Recall</div><div class="value" style="color:${NEON.amber}">${data.model_metrics.recall.toFixed(3)}</div></div>
    <div class="stat-card"><div class="label">ROC-AUC</div><div class="value" style="color:${NEON.pink}">${data.model_metrics.roc_auc.toFixed(3)}</div></div>
  `;

  destroyChart('importance');
  charts['importance'] = new Chart(document.getElementById('importance-chart'), {
    type: 'bar',
    data: { labels: data.global_feature_importance.map(f => f.feature),
      datasets: [{ data: data.global_feature_importance.map(f => f.importance), backgroundColor: NEON.cyan }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } } }
  });

  destroyChart('roc');
  charts['roc'] = new Chart(document.getElementById('roc-chart'), {
    type: 'line',
    data: {
      labels: data.roc_curve.fpr.map(v => v.toFixed(2)),
      datasets: [
        { label: 'ROC curve', data: data.roc_curve.tpr, borderColor: NEON.green, pointRadius: 0, tension: .1 },
        { label: 'Random guess', data: data.roc_curve.fpr, borderColor: '#444', borderDash: [4,4], pointRadius: 0 },
      ]
    },
    options: { scales: { x: { title: { display: true, text: 'False Positive Rate' } }, y: { title: { display: true, text: 'True Positive Rate' } } } }
  });

  destroyChart('pr');
  charts['pr'] = new Chart(document.getElementById('pr-chart'), {
    type: 'line',
    data: {
      labels: data.pr_curve.recall.map(v => v.toFixed(2)),
      datasets: [{ label: 'Precision-Recall', data: data.pr_curve.precision, borderColor: NEON.pink, pointRadius: 0, tension: .1 }]
    },
    options: { scales: { x: { title: { display: true, text: 'Recall' } }, y: { title: { display: true, text: 'Precision' } } } }
  });

  destroyChart('probhist');
  const edges = data.probability_histogram.edges;
  charts['probhist'] = new Chart(document.getElementById('prob-hist-chart'), {
    type: 'bar',
    data: {
      labels: edges.slice(0,-1).map((e,i) => `${e.toFixed(2)}-${edges[i+1].toFixed(2)}`),
      datasets: [{ data: data.probability_histogram.counts, backgroundColor: NEON.amber }]
    },
    options: { plugins: { legend: { display: false } }, scales: { y: { type: 'logarithmic', title: { display: true, text: 'Count (log)' } } } }
  });

  destroyChart('failtype');
  const ftColors = [NEON.cyan, NEON.pink, NEON.amber, NEON.purple, NEON.blue];
  charts['failtype'] = new Chart(document.getElementById('failure-type-chart'), {
    type: 'bar',
    data: {
      labels: Object.keys(data.failure_type_counts),
      datasets: [{ data: Object.values(data.failure_type_counts), backgroundColor: ftColors }]
    },
    options: { plugins: { legend: { display: false } } }
  });

  destroyChart('envcorr');
  charts['envcorr'] = new Chart(document.getElementById('env-corr-chart'), {
    type: 'bar',
    data: {
      labels: data.env_correlation.map(e => e.field),
      datasets: [{ data: data.env_correlation.map(e => e.correlation),
        backgroundColor: data.env_correlation.map(e => e.correlation >= 0 ? NEON.red : NEON.blue) }]
    },
    options: { indexAxis: 'y', plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Correlation with predicted risk' } } } }
  });
}

refreshFleet();
setInterval(refreshFleet, 4000);
setInterval(() => { if (document.getElementById('tab-detail').classList.contains('active')) loadMachineDetail(); }, 4000);
