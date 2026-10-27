// Optimal-control visualiser: bang-bang dosing against protein aggregation,
// following Sec. "Optimal therapeutic strategy design for protein aggregation" of the review
// and OptimalControl/aggregation.ipynb.
//   ẏ = α[u] + κ[u] y,   J = ∫₀ᵀ (y + ζ u) dt,   u(t) ∈ {0, u_max}, dosing only for T₁ < t < T₂.
//   primary-targeting drug:   α[u] = α₀ / (1 + K u)^n,    κ = κ₀
//   secondary-targeting drug: κ[u] = κ₀ / (1 + K u)^(n/2), α = α₀
// With u piecewise constant the kinetics is linear, so every run is integrated exactly.
// Units as in the notebook: t in h, y in nM, u in μM, K in 1/μM (1/μM = 10⁻³/nM).

(function () {
  "use strict";

  // ======================= model =======================

  const DEFAULTS = {
    primary: { a0: 0.04, k0: 10, n: 2, K: 1, umax: 4, zeta: 50, T: 2, T1: 0, T2: 0.5 },
    secondary: { a0: 0.04, k0: 10, n: 2, K: 1, umax: 4, zeta: 50, T: 2, T1: 0.2, T2: 1.0 },
  };

  function rates(drug, p, u) {
    if (drug === "primary") return { a: p.a0 / (1 + p.K * u) ** p.n, k: p.k0 };
    return { a: p.a0, k: p.k0 / (1 + p.K * u) ** (p.n / 2) };
  }

  // State and accumulated ∫y dt after time τ at constant rates, from y₀.
  function advance(y0, a, k, tau) {
    if (Math.abs(k * tau) < 1e-9) return { y: y0 + a * tau, Iy: y0 * tau + 0.5 * a * tau * tau };
    const e = Math.expm1(k * tau);
    const c = y0 + a / k;
    return { y: y0 + c * e, Iy: (c * e) / k - (a * tau) / k };
  }

  // the dosing window actually applied: ordered and inside [0, T]
  function windowOf(p, T1 = p.T1, T2 = p.T2) {
    const clamp = (x) => Math.min(Math.max(x, 0), p.T);
    const a = clamp(Math.min(T1, T2)), b = clamp(Math.max(T1, T2));
    return [a, b];
  }

  function segments(drug, p, T1, T2) {
    const [a, b] = windowOf(p, T1, T2);
    return [[0, a, 0], [a, b, p.umax], [b, p.T, 0]]
      .filter(([s, e]) => e > s)
      .map(([s, e, u]) => ({ s, e, u, ...rates(drug, p, u) }));
  }

  // total cost only (used for the landscape)
  function cost(drug, p, T1, T2) {
    let y = 0, Iy = 0, Iu = 0;
    for (const g of segments(drug, p, T1, T2)) {
      const r = advance(y, g.a, g.k, g.e - g.s);
      y = r.y; Iy += r.Iy; Iu += g.u * (g.e - g.s);
    }
    return { J: Iy + p.zeta * Iu, yT: y, Iy, Iu };
  }

  // sampled trajectory, with the switching times included exactly
  function simulate(drug, p, T1, T2) {
    const segs = segments(drug, p, T1, T2);
    const N = 400;
    const out = { t: [], y: [], u: [], J: [] };
    let y = 0, Iy = 0, Iu = 0;
    for (const g of segs) {
      const n = Math.max(2, Math.ceil((N * (g.e - g.s)) / p.T));
      for (let i = 0; i <= n; i++) {
        const tau = ((g.e - g.s) * i) / n;
        const r = advance(y, g.a, g.k, tau);
        out.t.push(g.s + tau);
        out.y.push(r.y);
        out.u.push(g.u);
        out.J.push(Iy + r.Iy + p.zeta * (Iu + g.u * tau));
      }
      const r = advance(y, g.a, g.k, g.e - g.s);
      y = r.y; Iy += r.Iy; Iu += g.u * (g.e - g.s);
    }
    out.metrics = { yT: y, Iy, drug: p.zeta * Iu, J: Iy + p.zeta * Iu };
    return out;
  }

  // Analytical switching times from the review (Michaels et al. 2019), clipped to [0, T].
  function optimalWindow(drug, p) {
    const { a, k } = rates(drug, p, p.umax);
    const clip = (x) => Math.min(Math.max(x, 0), p.T);
    if (drug === "primary") {
      // T₁ = 0,  T₂ = T − ln(ζ u_max κ₀ / (α₀ − α)) / κ₀
      const gain = p.a0 - a;
      const T2 = gain > 0 ? p.T - Math.log((p.zeta * p.umax * p.k0) / gain) / p.k0 : -Infinity;
      return { T1: 0, T2: clip(T2), raw: { T2 }, a, k };
    }
    // ΔT = κ₀/(κ₀ − κ) [T − ln(ζ u_max κ₀² / (α₀ (κ₀ − κ))) / κ₀],  window centred in [0, T]
    const gain = p.k0 - k;
    const dT = gain > 0
      ? (p.k0 / gain) * (p.T - Math.log((p.zeta * p.umax * p.k0 * p.k0) / (p.a0 * gain)) / p.k0)
      : -Infinity;
    const d = clip(dT);
    return { T1: (p.T - d) / 2, T2: (p.T + d) / 2, raw: { dT }, a, k };
  }

  // J(T₁, T₂) on a grid (log₁₀ for colour, raw for hover)
  function landscape(drug, p, n = 101) {
    const ax = Array.from({ length: n }, (_, i) => (p.T * i) / (n - 1));
    const z = [], J = [];
    for (let j = 0; j < n; j++) {
      const rowZ = [], rowJ = [];
      for (let i = 0; i < n; i++) {
        const c = ax[i] > ax[j] ? null : cost(drug, p, ax[i], ax[j]).J;
        rowZ.push(c === null ? null : Math.log10(c));
        rowJ.push(c);
      }
      z.push(rowZ);
      J.push(rowJ);
    }
    return { ax, z, J };
  }

  function run(drug, p) {
    return { sim: simulate(drug, p, p.T1, p.T2), params: p };
  }

  // ======================= sidebar specs =======================

  const SPEC = {
    kinetics: (drug) => [
      { key: "a0", label: "\\alpha_0", unit: "nM/h", min: 0.001, max: 0.2, step: 0.001, hint: "primary nucleation rate without drug" },
      { key: "k0", label: "\\kappa_0", unit: "1/h", min: 1, max: 20, step: 0.1, hint: "secondary nucleation rate without drug" },
      drug === "primary"
        ? { key: "n", label: "n_1", min: 0.5, max: 4, step: 0.5, hint: "reaction order of primary nucleation" }
        : { key: "n", label: "n_2", min: 0.5, max: 4, step: 0.5, hint: "reaction order of secondary nucleation" },
      {
        key: "K", label: drug === "primary" ? "K^{\\text{eq}}_1" : "K^{\\text{eq}}_2", unit: "1/μM", min: 0, max: 5, step: 0.01,
        hint: "drug–monomer binding constant (1 per μM is 0.001 per nM)",
      },
    ],
    cost: [
      { key: "umax", label: "u_{\\text{max}}", unit: "μM", min: 0.1, max: 10, step: 0.1, hint: "dose while treating" },
      { key: "zeta", label: "\\zeta", min: 0, max: 200, step: 1, hint: "toxicity of the drug relative to the aggregate" },
      { key: "T", label: "T", unit: "h", min: 0.5, max: 4, step: 0.05, hint: "treatment horizon" },
    ],
    schedule: [
      { key: "T1", label: "T_1", unit: "h", min: 0, max: 2, step: 0.005, hint: "dosing starts" },
      { key: "T2", label: "T_2", unit: "h", min: 0, max: 2, step: 0.005, hint: "dosing stops" },
    ],
  };

  // ======================= state =======================

  const DRUGS = ["primary", "secondary"];
  const STORE_KEY = "cb.oc.v1";
  const saved0 = CB.store.get(STORE_KEY, null);
  const state = {
    drug: "primary",
    params: { primary: { ...DEFAULTS.primary }, secondary: { ...DEFAULTS.secondary } },
    showOpen: true,
    showOpt: true,
    logScale: true,
    saved: { primary: [], secondary: [] },
  };
  if (saved0) {
    state.drug = saved0.drug === "secondary" ? "secondary" : "primary";
    for (const d of DRUGS) {
      state.params[d] = { ...DEFAULTS[d], ...(saved0.params?.[d] || {}) };
      state.saved[d] = Array.isArray(saved0.saved?.[d]) ? saved0.saved[d] : [];
    }
    for (const k of ["showOpen", "showOpt", "logScale"]) state[k] = saved0[k] !== false;
  }
  readHash();

  function persist() {
    const { drug, params, saved, showOpen, showOpt, logScale } = state;
    CB.store.set(STORE_KEY, { drug, params, saved, showOpen, showOpt, logScale });
  }

  // share links: #drug=primary&T1=0&...
  function readHash() {
    if (!location.hash || location.hash.length < 2) return;
    const q = new URLSearchParams(location.hash.slice(1));
    const d = q.get("drug");
    if (!DRUGS.includes(d)) return;
    state.drug = d;
    const p = { ...DEFAULTS[d] };
    for (const [k, v] of q.entries()) if (k in p) p[k] = parseFloat(v);
    state.params[d] = p;
  }

  function shareLink() {
    const q = new URLSearchParams({ drug: state.drug });
    for (const [k, v] of Object.entries(state.params[state.drug])) q.set(k, String(v));
    return location.href.split("#")[0] + "#" + q.toString();
  }

  // ======================= sidebar =======================

  const sidebar = document.getElementById("sidebar");
  const { heading, segmented, button } = CB;
  let scheduleCtrls = []; // T₁, T₂ sliders, whose range follows T
  let derivedBox = null;
  let savedListEl = null;

  function onParam(key, value) {
    const p = state.params[state.drug];
    p[key] = value;
    if (key === "T") syncScheduleRange();
    scheduleDraw();
  }

  function syncScheduleRange() {
    const T = state.params[state.drug].T;
    for (const c of scheduleCtrls) c.el.querySelector('input[type="range"]').max = T;
  }

  const sliders = (specs, p) => specs.map((s) => CB.slider(s, p[s.key], onParam));

  function buildSidebar() {
    const d = state.drug;
    const p = state.params[d];
    sidebar.innerHTML = "";

    sidebar.append(heading("Nucleation targeted by the drug"));
    sidebar.append(
      segmented([["primary", "Primary"], ["secondary", "Secondary"]], d, (v) => {
        state.drug = v;
        buildSidebar();
        scheduleDraw();
      })
    );

    sidebar.append(heading("Aggregation kinetics"), ...sliders(SPEC.kinetics(d), p).map((c) => c.el));
    sidebar.append(heading("Dose and cost"), ...sliders(SPEC.cost, p).map((c) => c.el));

    sidebar.append(heading("Optimal schedule"));
    derivedBox = document.createElement("div");
    derivedBox.className = "derived";
    sidebar.append(derivedBox);

    sidebar.append(heading("Your schedule"));
    scheduleCtrls = sliders(SPEC.schedule, p);
    sidebar.append(...scheduleCtrls.map((c) => c.el));
    syncScheduleRange();
    const rowS = document.createElement("div");
    rowS.className = "btn-row";
    rowS.append(
      button("Use the optimal window", () => {
        const o = optimalWindow(d, p);
        p.T1 = round4(o.T1);
        p.T2 = round4(o.T2);
        buildSidebar();
        scheduleDraw();
      }),
      button("No drug", () => {
        p.T1 = 0;
        p.T2 = 0;
        buildSidebar();
        scheduleDraw();
      })
    );
    sidebar.append(rowS);

    sidebar.append(heading("Display"));
    for (const [key, label] of [["showOpt", "Show optimal schedule"], ["showOpen", "Show uncontrolled run"], ["logScale", "Log scale for aggregate and cost"]]) {
      sidebar.append(CB.checkbox({ key, label }, state[key], (k, v) => { state[k] = v; scheduleDraw(); }).el);
    }

    sidebar.append(heading("Saved parameter sets"));
    const row = document.createElement("div");
    row.className = "btn-row";
    row.append(
      button("Save current", saveCurrent, "primary"),
      button("Export", exportSets),
      button("Import", importSets),
      button("Clear", () => {
        if (state.saved[d].length && confirm("Delete all saved sets for this drug?")) {
          state.saved[d] = [];
          scheduleDraw();
        }
      })
    );
    sidebar.append(row);
    savedListEl = document.createElement("ul");
    savedListEl.className = "saved-list";
    sidebar.append(savedListEl);

    sidebar.append(heading("Page"));
    const row2 = document.createElement("div");
    row2.className = "btn-row";
    row2.append(
      button("Copy link to these parameters", async (ev) => {
        const link = shareLink();
        history.replaceState(null, "", link);
        try {
          await navigator.clipboard.writeText(link);
          ev.target.textContent = "Link copied";
        } catch (e) {
          ev.target.textContent = "Link is in the address bar";
        }
        setTimeout(() => (ev.target.textContent = "Copy link to these parameters"), 1600);
      }),
      button("Reset parameters", () => {
        state.params[d] = { ...DEFAULTS[d] };
        buildSidebar();
        scheduleDraw();
      })
    );
    sidebar.append(row2);
  }

  const round4 = (x) => Math.round(x * 1e4) / 1e4;

  // ======================= saved sets =======================

  // hues of the current run and the optimal schedule, kept clear of saved colours
  const reservedHues = () => [{ h: 222 }, { h: 28 }];

  const windowText = (p) => {
    const [a, b] = windowOf(p);
    return b > a ? `${CB.fmt(a)}–${CB.fmt(b)} h` : "no drug";
  };

  function summary(p) {
    return `${windowText(p)} · umax=${CB.fmt(p.umax)} · ζ=${CB.fmt(p.zeta)} · Keq=${CB.fmt(p.K)} · T=${CB.fmt(p.T)}`;
  }

  function saveCurrent() {
    CB.addSavedSet(state.saved[state.drug], state.params[state.drug], reservedHues());
    scheduleDraw();
  }

  function renderSavedList() {
    const d = state.drug;
    CB.renderSavedList(savedListEl, state.saved[d], {
      onChange: scheduleDraw,
      onLoad: (s) => {
        state.params[d] = { ...DEFAULTS[d], ...s.params };
        buildSidebar();
        scheduleDraw();
      },
      summary: (s) => summary(s.params),
      reserved: reservedHues(),
    });
  }

  function exportSets() {
    CB.download(`oc_saved_sets_${state.drug}.json`, JSON.stringify({ drug: state.drug, saved: state.saved[state.drug] }, null, 2));
  }

  function importSets() {
    CB.importJson((data) => {
      const d = data.drug === "secondary" ? "secondary" : "primary";
      for (const s of data.saved || []) CB.addSavedSet(state.saved[d], { ...DEFAULTS[d], ...s.params }, reservedHues()).name = s.name || "Imported";
      state.drug = d;
      buildSidebar();
      scheduleDraw();
    });
  }

  // ======================= plots =======================

  const { PHI, axisStyle, baseLayout } = CB;
  const mainPlot = document.getElementById("plot-main");
  const landPlot = document.getElementById("plot-landscape");
  const legendEl = document.getElementById("legend");
  const metricsEl = document.getElementById("metrics");

  const scheduleDraw = CB.frameScheduler(() => draw());

  function draw() {
    const d = state.drug;
    const p = state.params[d];
    const cur = CB.cssVar("--current");
    const optC = CB.cssVar("--optimal");
    const muted = CB.cssVar("--muted");

    const opt = optimalWindow(d, p);
    const runs = [];
    for (const s of state.saved[d]) if (s.visible) runs.push({ name: s.name, colour: s.colour.css, width: 1.6, ...run(d, { ...DEFAULTS[d], ...s.params }) });
    runs.push({ name: "Current", colour: cur, width: 2.6, current: true, ...run(d, p) });
    const optimal = { name: "Optimal", colour: optC, width: 2.2, dash: "dash", sim: simulate(d, p, opt.T1, opt.T2), params: { ...p, T1: opt.T1, T2: opt.T2 } };
    const open = { name: "Uncontrolled", colour: muted, width: 1.4, dash: "dot", sim: simulate(d, p, 0, 0), params: { ...p, T1: 0, T2: 0 } };

    document.body.dataset.drug = d;
    const land = landscape(d, p);
    renderDerived(d, p, opt);
    renderSavedList();
    persist();

    // ----- time series: dose, aggregate, accumulated cost -----
    const shown = [...(state.showOpen ? [open] : []), ...(state.showOpt ? [optimal] : []), ...runs];
    const traces = [];
    const add = (r, key, axis, extra = {}) =>
      traces.push({
        x: r.sim.t, y: r.sim[key], xaxis: "x", yaxis: axis, name: r.name, type: "scatter", mode: "lines",
        line: { color: r.colour, width: r.width, dash: r.dash || "solid", ...(extra.line || {}) },
        hovertemplate: "%{y:.4g}", ...extra.trace,
      });
    for (const r of shown) {
      if (r !== open) add(r, "u", "y", r.current ? { trace: { fill: "tozeroy", fillcolor: CB.cssVar("--current-fill") } } : {});
      add(r, "y", "y2");
      add(r, "J", "y3");
    }
    const uMax = Math.max(...shown.map((r) => r.params.umax));
    const yType = state.logScale ? "log" : "linear";
    const W = mainPlot.clientWidth;
    const layout = {
      ...baseLayout(),
      width: W,
      height: Math.max(460, W / PHI), // golden-ratio panel
      margin: { l: 70, r: 16, t: 8, b: 44 },
      xaxis: axisStyle({ anchor: "y3", title: { text: "<i>t</i> (h)" }, range: [0, p.T] }),
      yaxis: axisStyle({ domain: [0.76, 1], title: { text: "Dose <i>u</i>(<i>t</i>)<br>(μM)" }, range: [0, uMax * 1.15] }),
      yaxis2: axisStyle({ domain: [0.4, 0.71], type: yType, exponentformat: "power", title: { text: "Aggregate <i>y</i>(<i>t</i>)<br>(nM)" } }),
      yaxis3: axisStyle({ domain: [0, 0.35], type: yType, exponentformat: "power", title: { text: "Accumulated cost<br>∫(<i>y</i> + <i>ζu</i>) d<i>t</i>" } }),
    };
    Plotly.react(mainPlot, traces, layout, CB.PLOT_CONFIG);

    CB.renderLegend(legendEl, [
      ...runs.map((r) => [r.colour, r.name]),
      ...(state.showOpt ? [[optC, "Optimal (analytical)", true]] : []),
      ...(state.showOpen ? [[muted, "Uncontrolled", "dot"]] : []),
    ]);

    drawLandscape(p, land, opt, runs);
    renderMetrics([...runs].reverse().concat([optimal, open]));
  }

  function drawLandscape(p, land, opt, runs) {
    const panel = CB.cssVar("--panel");
    const marker = (x, y, name, colour, symbol, size) => ({
      x: [x], y: [y], mode: "markers", type: "scatter", name,
      marker: { symbol, size, color: colour, line: { width: 1.5, color: panel } },
      hovertemplate: `${name}<br>T₁ %{x:.3f} h, T₂ %{y:.3f} h<extra></extra>`,
    });
    const traces = [{
      type: "heatmap", x: land.ax, y: land.ax, z: land.z, customdata: land.J,
      colorscale: "Viridis", zsmooth: "best", hoverongaps: false,
      colorbar: { title: { text: "log<sub>10</sub> total cost", side: "right" }, thickness: 12, outlinewidth: 0 },
      hovertemplate: "T₁ %{x:.3f} h, T₂ %{y:.3f} h<br>total cost %{customdata:.4g}<extra></extra>",
    }];
    for (const r of runs) {
      if (r.params.T !== p.T && !r.current) continue; // a different horizon is a different landscape
      const [a, b] = windowOf(r.params);
      traces.push(marker(a, b, r.name, r.colour, "circle", r.current ? 12 : 9));
    }
    traces.push(marker(opt.T1, opt.T2, "Optimal (analytical)", CB.cssVar("--optimal"), "star", 15));
    const layout = {
      ...baseLayout(), hovermode: "closest", margin: { l: 56, r: 12, t: 6, b: 44 },
      width: landPlot.clientWidth, height: landPlot.clientHeight,
      xaxis: axisStyle({ title: { text: "<i>T</i><sub>1</sub> (h)" }, range: [0, p.T], showgrid: false, zeroline: false, scaleanchor: "y", constrain: "domain" }),
      yaxis: axisStyle({ title: { text: "<i>T</i><sub>2</sub> (h)" }, range: [0, p.T], showgrid: false, zeroline: false }),
    };
    Plotly.react(landPlot, traces, layout, CB.PLOT_CONFIG);
  }

  function renderMetrics(rows) {
    const cols = [
      ["dosing window", (r) => windowText(r.params)],
      ["y(T) (nM)", (r) => r.sim.metrics.yT],
      ["∫y dt", (r) => r.sim.metrics.Iy],
      ["ζ∫u dt", (r) => r.sim.metrics.drug],
      ["total cost", (r) => r.sim.metrics.J],
    ];
    let html = "<table class='metrics'><thead><tr><th>Set</th>" + cols.map((c) => `<th>${c[0]}</th>`).join("") + "</tr></thead><tbody>";
    for (const r of rows) {
      html += `<tr><td><span class="dot" style="background:${r.colour}"></span>${CB.escapeHtml(r.name)}</td>`;
      for (const [, f] of cols) {
        const v = f(r);
        html += typeof v === "number" ? `<td class="num">${CB.fmt(v, 4)}</td>` : `<td>${v}</td>`;
      }
      html += "</tr>";
    }
    metricsEl.innerHTML = html + "</tbody></table>";
  }

  function renderDerived(d, p, opt) {
    const rows = [];
    if (d === "primary") {
      rows.push(["\\alpha(u_{\\text{max}})", CB.fmt(opt.a, 4) + " nM/h"]);
      rows.push(["T_1", "0 h"]);
      rows.push(["T_2", CB.fmt(opt.T2, 4) + " h"]);
      if (!(opt.raw.T2 > 0)) rows.push(["", "no drug is optimal"]);
    } else {
      rows.push(["\\kappa(u_{\\text{max}})", CB.fmt(opt.k, 4) + " 1/h"]);
      rows.push(["\\Delta T", CB.fmt(opt.raw.dT, 4) + " h"]);
      rows.push(["T_1,\\ T_2", `${CB.fmt(opt.T1, 4)}, ${CB.fmt(opt.T2, 4)} h`]);
      if (!(opt.raw.dT > 0)) rows.push(["", "no drug is optimal"]);
      else if (opt.raw.dT > p.T) rows.push(["", "dose throughout"]);
    }
    rows.push(["\\text{total cost}", CB.fmt(cost(d, p, opt.T1, opt.T2).J, 4)]);
    const texOf = (k) => (window.katex ? katex.renderToString(k, { throwOnError: false }) : k);
    derivedBox.innerHTML = rows.map(([k, v]) => `<div><span>${k ? texOf(k) : ""}</span><span class="v">${v}</span></div>`).join("");
  }

  // ======================= boot =======================

  function boot() {
    CB.renderTabs("oc");
    CB.renderPageMath(document.querySelector(".main"));
    buildSidebar();
    draw();
    window.addEventListener("themechange", scheduleDraw);
    CB.onWidthChange([mainPlot, landPlot], scheduleDraw);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
