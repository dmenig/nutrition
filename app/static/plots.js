function sliceRange(data, range) {
  if (range === 'all') return data;
  const n = parseInt(range, 10);
  return data.slice(Math.max(0, data.length - n));
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

function toSeriesXY(series) {
  return series.map((p) => {
    const t = Number(p.time_index);
    // Normalize to milliseconds: ECharts expects ms. If backend sent seconds,
    // values will be ~1e9; convert to ms. If already ms (~1e12+), keep as-is.
    const ms = t < 1e12 ? t * 1000 : t;
    return [ms, p.value];
  });
}

async function renderWeightChart(chart, range) {
  const data = await fetchJSON('/api/v1/plots/weight');
  const wObs = sliceRange(toSeriesXY(data.W_obs), range);
  const wAdj = sliceRange(toSeriesXY(data.W_adj_pred), range);
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['Observed', 'Adjusted'] },
    xAxis: { type: 'time', name: 'Date' },
    yAxis: { type: 'value', name: 'Weight (kg)' },
    series: [
      { name: 'Observed', type: 'line', showSymbol: false, data: wObs },
      { name: 'Adjusted', type: 'line', showSymbol: false, data: wAdj },
    ],
  });
}

async function renderMetabolismChart(chart, range) {
  const data = await fetchJSON('/api/v1/plots/metabolism');
  const mBase = sliceRange(toSeriesXY(data.M_base), range);
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['Base Metabolism'] },
    xAxis: { type: 'time', name: 'Date' },
    yAxis: { type: 'value', name: 'kcal/day' },
    series: [{ name: 'Base Metabolism', type: 'line', showSymbol: false, data: mBase }],
  });
}

async function renderEnergyChart(chart, range) {
  const data = await fetchJSON('/api/v1/plots/energy-balance');
  const cal = sliceRange(toSeriesXY(data.calories_unnorm), range);
  const exp = sliceRange(toSeriesXY(data.C_exp_t), range);
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['Calories In', 'Expended (metabolism + sport)'] },
    xAxis: { type: 'time', name: 'Date' },
    yAxis: { type: 'value', name: 'kcal' },
    series: [
      { name: 'Calories In', type: 'line', showSymbol: false, data: cal },
      { name: 'Expended (metabolism + sport)', type: 'line', showSymbol: false, data: exp },
    ],
  });
}

async function render() {
  const chart = echarts.init(document.getElementById('chart'));
  const plotSel = document.getElementById('plot-selector');
  const rangeSel = document.getElementById('range-selector');
  const refreshBtn = document.getElementById('refresh-btn');

  async function go() {
    const kind = plotSel.value;
    const range = rangeSel.value;
    if (kind === 'weight') await renderWeightChart(chart, range);
    else if (kind === 'metabolism') await renderMetabolismChart(chart, range);
    else await renderEnergyChart(chart, range);
  }

  window.addEventListener('resize', () => chart.resize());
  plotSel.addEventListener('change', go);
  rangeSel.addEventListener('change', go);
  refreshBtn.addEventListener('click', go);
  await go();
}

render().catch((e) => {
  console.error(e);
  alert('Failed to load plots. Check console.');
});



