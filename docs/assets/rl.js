// Reinforcement-learning visualiser: adaptive cancer therapy, following Sec.
// "Learning tumor drug dosing strategies" of the review and
// ReinforcementLearning/adaptive_therapy.ipynb.
//   Ṡ = r_S S (1 − N/K)(1 − d_D D) − d_S S,   Ṙ = r_R R (1 − N/K) − d_R R,   N = S + R,
//   progression when N ≥ (1 + κ) N(0); the dose D ∈ {0, 1} is chosen weekly from N alone.
// The agents are pre-trained offline (docs/rl/train_agents.py → docs/rl/agents.js); this page
// only evaluates their policy networks. The integrator (RK4, fixed steps per week) and reward
// are the training script's, so the agents' decisions are reproduced exactly.

(function () {
  "use strict";

  const AG = window.RL_AGENTS;
  const DAYS_PER_MONTH = 30.4;
  const KAPPA = AG.reward.progression_factor - 1;
  const HORIZON = AG.reward.horizon_days;

  // ======================= model =======================

  const PATIENT_KEYS = ["rS", "rR", "dS", "dR", "dD", "S0", "R0"];
  const presetById = (id) => AG.presets.find((p) => p.id === id) || AG.presets[0];

  function defaultsFor(id) {
    const pr = presetById(id);
    const p = { preset: pr.id, on: 1.1, off: 0.9 };
    for (const k of PATIENT_KEYS) p[k] = pr.patient[k];
    return p;
  }

  function rhs(p, S, R, D) {
    const comp = 1 - (S + R); // K = 1: sizes are fractions of the carrying capacity
    return [p.rS * comp * (1 - p.dD * D) * S - p.dS * S, p.rR * comp * R - p.dR * R];
  }

  // Greedy forward pass of a policy network; returns the logits.
  function logits(layers, x) {
    for (let k = 0; k < layers.length; k++) {
      const L = layers[k];
      const [nOut, nIn] = L.shape;
      const y = new Array(nOut);
      for (let i = 0; i < nOut; i++) {
        let s = L.b[i];
        for (let j = 0; j < nIn; j++) s += L.W[i * nIn + j] * x[j];
        y[i] = k < layers.length - 1 ? Math.max(s, 0) : s;
      }
      x = y;
    }
    return x;
  }
  const pTreat = (layers, size, D) => {
    const [l0, l1] = logits(layers, [size, D]);
    return 1 / (1 + Math.exp(l0 - l1));
  };

  // policies: (size = N/N(0), D = dose of the past week) -> 0 holiday / 1 treat
  const POLICY = {
    ct: () => 1,
    at50: (size, D) => (D > 0 ? (size <= 0.5 ? 0 : 1) : size >= 1 ? 1 : 0),
    rl: (layers) => (size, D) => { const l = logits(layers, [size, D]); return l[1] > l[0] ? 1 : 0; },
    rule: (on, off) => (size, D) => (D > 0 ? (size <= off ? 0 : 1) : size >= on ? 1 : 0),
  };

  // One patient under one policy, until progression or the 20-year horizon.
  function simulate(p, policy) {
    const steps = AG.rk4_steps, h = AG.dt_days / steps;
    const n0 = p.S0 + p.R0;
    const out = { t: [0], S: [1 * p.S0 / n0], R: [p.R0 / n0], N: [1], dT: [], dose: [] };
    let S = p.S0, R = p.R0, D = 0, t = 0, reward = 0, treated = 0, holidays = 0, ttp = null;
    for (;;) {
      const size = (S + R) / n0;
      const a = policy(size, D);
      if (a === 0 && D > 0) holidays++;
      D = a;
      out.dT.push(t / DAYS_PER_MONTH);
      out.dose.push(D);
      for (let s = 0; s < steps; s++) {
        const k1 = rhs(p, S, R, D);
        const k2 = rhs(p, S + 0.5 * h * k1[0], R + 0.5 * h * k1[1], D);
        const k3 = rhs(p, S + 0.5 * h * k2[0], R + 0.5 * h * k2[1], D);
        const k4 = rhs(p, S + h * k3[0], R + h * k3[1], D);
        S += (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        R += (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        out.t.push((t + (s + 1) * h) / DAYS_PER_MONTH);
        out.S.push(S / n0);
        out.R.push(R / n0);
        out.N.push((S + R) / n0);
      }
      t += AG.dt_days;
      if (D > 0) treated += AG.dt_days;
      // the notebook's reward
      const N = S + R;
      if (N > AG.reward.progression_factor * n0) { reward += AG.reward.punish; ttp = t; break; }
      let r = AG.reward.base;
      if (t >= HORIZON) { reward += r + AG.reward.survival_bonus; break; }
      if (a === 0) r += (N > n0 ? AG.reward.punish : 0) + AG.reward.hol;
      reward += r;
    }
    out.dT.push(t / DAYS_PER_MONTH);
    out.dose.push(D);
    out.metrics = { ttp, end: t, treatedFrac: treated / t, holidays, reward };
    return out;
  }

  const isPreset = (p) => PATIENT_KEYS.every((k) => p[k] === presetById(p.preset).patient[k]);

  // ======================= sidebar specs =======================

  const SPEC = {
    patient: [
      { key: "rS", label: "r_S", unit: "1/day", min: 0.005, max: 0.06, step: 0.001, hint: "proliferation rate of sensitive cells" },
      { key: "rR", label: "r_R", unit: "1/day", min: 0.005, max: 0.06, step: 0.001, hint: "proliferation rate of resistant cells" },
      { key: "dS", label: "d_S", unit: "1/day", min: 0, max: 0.005, step: 0.0001, hint: "death rate of sensitive cells" },
      { key: "dR", label: "d_R", unit: "1/day", min: 0, max: 0.005, step: 0.0001, hint: "death rate of resistant cells" },
      { key: "dD", label: "d_D", min: 0, max: 3, step: 0.01, hint: "drug kill factor on sensitive cells" },
      { key: "S0", label: "S(0)", min: 0.05, max: 0.85, step: 0.005, hint: "initial sensitive population, as a fraction of the carrying capacity" },
      { key: "R0", label: "R(0)", min: 0.001, max: 0.3, step: 0.001, hint: "initial resistant population, as a fraction of the carrying capacity" },
    ],
    rule: [
      { key: "on", label: "\\text{treat from }N/N(0)", min: 0.3, max: 1.2, step: 0.01, hint: "treatment starts when the tumour grows to this size" },
      { key: "off", label: "\\text{pause at }N/N(0)", min: 0, max: 1.2, step: 0.01, hint: "treatment stops when the tumour shrinks to this size" },
    ],
  };

  // ======================= state =======================

  const STORE_KEY = "cb.rl.v1";
  const saved0 = CB.store.get(STORE_KEY, null);
  const state = { params: defaultsFor("reference"), showSR: true, saved: [] };
  if (saved0) {
    state.params = { ...defaultsFor(saved0.params?.preset), ...(saved0.params || {}) };
    state.params.preset = presetById(state.params.preset).id;
    state.saved = Array.isArray(saved0.saved) ? saved0.saved : [];
    state.showSR = saved0.showSR !== false;
  }
  readHash();

  function persist() {
    CB.store.set(STORE_KEY, { params: state.params, saved: state.saved, showSR: state.showSR });
  }

  function readHash() {
    if (!location.hash || location.hash.length < 2) return;
    const q = new URLSearchParams(location.hash.slice(1));
    if (!q.has("preset")) return;
    const p = defaultsFor(q.get("preset"));
    for (const [k, v] of q.entries()) if (k in p && k !== "preset") p[k] = parseFloat(v);
    state.params = p;
  }

  function shareLink() {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(state.params)) q.set(k, String(v));
    return location.href.split("#")[0] + "#" + q.toString();
  }

  // ======================= sidebar =======================

  const sidebar = document.getElementById("sidebar");
  const { heading, button } = CB;
  let derivedBox = null;
  let savedListEl = null;

  function onParam(key, value) {
    state.params[key] = value;
    scheduleDraw();
  }

  const sliders = (specs, p) => specs.map((s) => CB.slider(s, p[s.key], onParam).el);

  function buildSidebar() {
    const p = state.params;
    sidebar.innerHTML = "";

    sidebar.append(heading("Virtual patient"));
    sidebar.append(
      CB.select({ key: "preset", label: "Preset (each has its own trained agent)", options: AG.presets.map((pr) => [pr.id, pr.name]) }, p.preset, (k, v) => {
        state.params = { ...defaultsFor(v), on: p.on, off: p.off };
        buildSidebar();
        scheduleDraw();
      }).el
    );
    const desc = document.createElement("p");
    desc.className = "hint";
    desc.textContent = presetById(p.preset).desc;
    sidebar.append(desc);
    derivedBox = document.createElement("div");
    derivedBox.className = "derived";
    sidebar.append(derivedBox);
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "Patient parameters";
    sum.style.cursor = "pointer";
    sum.style.margin = "6px 0";
    det.append(sum, ...sliders(SPEC.patient, p));
    const rowP = document.createElement("div");
    rowP.className = "btn-row";
    rowP.append(button("Back to the preset patient", () => {
      state.params = { ...defaultsFor(p.preset), on: p.on, off: p.off };
      buildSidebar();
      scheduleDraw();
    }));
    det.append(rowP);
    det.open = !isPreset(p);
    sidebar.append(det);

    sidebar.append(heading("Your threshold rule"));
    sidebar.append(...sliders(SPEC.rule, p));
    const rowR = document.createElement("div");
    rowR.className = "btn-row";
    rowR.append(button("Set to AT50", () => {
      Object.assign(p, { on: 1, off: 0.5 });
      buildSidebar();
      scheduleDraw();
    }));
    sidebar.append(rowR);

    sidebar.append(heading("Display"));
    sidebar.append(CB.checkbox({ key: "showSR", label: "Show sensitive and resistant cells" }, state.showSR, (k, v) => { state.showSR = v; scheduleDraw(); }).el);

    sidebar.append(heading("Saved parameter sets"));
    const row = document.createElement("div");
    row.className = "btn-row";
    row.append(
      button("Save current", saveCurrent, "primary"),
      button("Export", exportSets),
      button("Import", importSets),
      button("Clear", () => {
        if (state.saved.length && confirm("Delete all saved sets?")) {
          state.saved = [];
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
        state.params = defaultsFor("reference");
        buildSidebar();
        scheduleDraw();
      })
    );
    sidebar.append(row2);
  }

  // ======================= saved sets =======================

  const reservedHues = () => [{ h: 222 }];

  function summary(p) {
    const pr = presetById(p.preset);
    return `on ${CB.fmt(p.on)}, off ${CB.fmt(p.off)} · ${pr.name}${isPreset(p) ? "" : " (edited)"}`;
  }

  function saveCurrent() {
    CB.addSavedSet(state.saved, state.params, reservedHues());
    scheduleDraw();
  }

  function renderSavedList() {
    CB.renderSavedList(savedListEl, state.saved, {
      onChange: scheduleDraw,
      onLoad: (s) => {
        state.params = { ...defaultsFor(s.params.preset), ...s.params };
        buildSidebar();
        scheduleDraw();
      },
      summary: (s) => summary(s.params),
      reserved: reservedHues(),
    });
  }

  function exportSets() {
    CB.download("rl_saved_sets.json", JSON.stringify({ saved: state.saved }, null, 2));
  }

  function importSets() {
    CB.importJson((data) => {
      for (const s of data.saved || []) {
        CB.addSavedSet(state.saved, { ...defaultsFor(s.params?.preset), ...s.params }, reservedHues()).name = s.name || "Imported";
      }
      scheduleDraw();
    });
  }

  // ======================= plots =======================

  const { PHI, axisStyle, baseLayout } = CB;
  const mainPlot = document.getElementById("plot-main");
  const policyPlot = document.getElementById("plot-policy");
  const legendEl = document.getElementById("legend");
  const metricsEl = document.getElementById("metrics");

  const scheduleDraw = CB.frameScheduler(() => draw());

  function runsFor(p) {
    const layers = presetById(p.preset).policy;
    return [
      { name: "Continuous therapy", short: "CT", sim: simulate(p, POLICY.ct) },
      { name: "AT50", short: "AT50", sim: simulate(p, POLICY.at50) },
      { name: "Reinforcement learning", short: "RL", sim: simulate(p, POLICY.rl(layers)) },
    ];
  }

  function draw() {
    const p = state.params;
    const ink = CB.cssVar("--ink");
    const cS = CB.cssVar("--cell-s"), cR = CB.cssVar("--cell-r"), cDose = CB.cssVar("--dose");
    const [ct, at50, rl] = runsFor(p);
    ct.colour = at50.colour = rl.colour = ink;
    const blocks = [ct, at50, rl, { name: "Your rule", short: "Yours", colour: CB.cssVar("--current"), sim: simulate(p, POLICY.rule(p.on, p.off)), params: p }];
    for (const s of state.saved) {
      if (!s.visible) continue;
      const sp = { ...defaultsFor(s.params.preset), ...s.params };
      blocks.push({ name: s.name, short: s.name, colour: s.colour.css, sim: simulate(sp, POLICY.rule(sp.on, sp.off)), params: sp });
    }

    renderDerived(p, rl);
    renderSavedList();
    persist();

    // time window: up to the RL agent's time to progression, as in the review's figure
    const tEnd = rl.sim.metrics.end / DAYS_PER_MONTH;
    const xMax = tEnd * 1.06;
    const cut = (sim) => {
      let n = sim.t.findIndex((t) => t > xMax);
      if (n < 0) n = sim.t.length;
      return n;
    };

    // stacked blocks: a dose strip over the tumour size, one block per strategy
    const nb = blocks.length;
    const gapB = 0.045, gapIn = 0.03, doseFrac = 0.24;
    const H = (1 - gapB * (nb - 1)) / nb;
    const traces = [], shapes = [], annotations = [];
    const layout = {
      ...baseLayout(), hovermode: "closest",
      xaxis: axisStyle({ title: { text: "time (month)" }, range: [0, xMax], anchor: `y${2 * nb}`, showgrid: false }),
    };
    const yMaxN = Math.max(1.5, ...blocks.map((b) => Math.max(...b.sim.N.slice(0, cut(b.sim))) * 1.05));
    blocks.forEach((b, i) => {
      const top = 1 - i * (H + gapB);
      const dAx = 2 * i + 1, nAx = 2 * i + 2;
      const dName = dAx === 1 ? "y" : `y${dAx}`, nName = `y${nAx}`;
      layout[`yaxis${dAx === 1 ? "" : dAx}`] = axisStyle({
        domain: [top - doseFrac * H, top], range: [-0.15, 1.15], tickvals: [0, 1], ticktext: ["holiday", "treat"],
        showgrid: false, zeroline: false, fixedrange: true,
      });
      layout[`yaxis${nAx}`] = axisStyle({
        domain: [top - H, top - doseFrac * H - gapIn], range: [0, yMaxN], dtick: 0.5,
        title: { text: "tumour size" }, zeroline: false,
      });
      const n = cut(b.sim);
      const nd = b.sim.dT.findIndex((t) => t > xMax);
      const dT = nd < 0 ? b.sim.dT : b.sim.dT.slice(0, nd);
      const dose = nd < 0 ? b.sim.dose : b.sim.dose.slice(0, nd);
      traces.push({
        x: dT, y: dose, xaxis: "x", yaxis: dName, type: "scatter", mode: "lines", line: { shape: "hv", color: cDose, width: 1.6 },
        fill: "tozeroy", fillcolor: CB.cssVar("--dose-fill"), hovertemplate: `${b.short}: %{y}<extra></extra>`,
      });
      const line = (key, colour, width, label) =>
        traces.push({
          x: b.sim.t.slice(0, n), y: b.sim[key].slice(0, n), xaxis: "x", yaxis: nName, type: "scatter", mode: "lines",
          line: { color: colour, width }, hovertemplate: `${b.short} ${label} %{y:.3f}<br>t = %{x:.1f} month<extra></extra>`,
        });
      if (state.showSR) {
        line("S", cS, 1.6, "S/N(0)");
        line("R", cR, 1.6, "R/N(0)");
      }
      line("N", ink, 2.2, "N/N(0)");
      // progression threshold and the progressed region
      shapes.push({ type: "line", xref: "x", yref: nName, x0: 0, x1: xMax, y0: 1 + KAPPA, y1: 1 + KAPPA, line: { color: ink, width: 1.2, dash: "dash" } });
      const ttp = b.sim.metrics.ttp;
      if (ttp !== null && ttp / DAYS_PER_MONTH < xMax) {
        for (const ax of [dName, nName]) {
          shapes.push({
            type: "rect", xref: "x", yref: `${ax} domain`, x0: ttp / DAYS_PER_MONTH, x1: xMax, y0: 0, y1: 1,
            fillcolor: CB.cssVar("--progressed"), line: { width: 0 }, layer: "below",
          });
        }
      }
      annotations.push({
        xref: "paper", yref: "paper", x: 1.005, xanchor: "left", y: top - H / 2, yanchor: "middle",
        text: b.name, textangle: 90, showarrow: false, font: { color: b.colour },
      });
    });
    const W = mainPlot.clientWidth;
    Object.assign(layout, {
      width: W,
      height: Math.max(W / PHI, 165 * nb + 60),
      margin: { l: 70, r: 34, t: 8, b: 44 },
      shapes, annotations,
    });
    Plotly.react(mainPlot, traces, layout, CB.PLOT_CONFIG);

    CB.renderLegend(legendEl, [
      ...(state.showSR ? [[cS, "S(t)/N(0)"], [cR, "R(t)/N(0)"]] : []),
      [ink, "N(t)/N(0)"], [cDose, "dose"], [ink, "progression threshold", true], [CB.cssVar("--progressed-key"), "progressed", "shade"],
    ]);

    drawPolicy(p);
    renderMetrics(blocks);
  }

  function drawPolicy(p) {
    const layers = presetById(p.preset).policy;
    const cDose = CB.cssVar("--dose"), cur = CB.cssVar("--current");
    const xs = Array.from({ length: 261 }, (_, i) => (1.3 * i) / 260);
    const traces = [];
    for (const [D, dash, label] of [[1, "solid", "treating"], [0, "dash", "on holiday"]]) {
      traces.push({
        x: xs, y: xs.map((x) => pTreat(layers, x, D)), type: "scatter", mode: "lines",
        line: { color: cDose, width: 2.2, dash }, name: `agent, ${label}`,
        hovertemplate: `agent, ${label}<br>N/N(0) %{x:.2f}: P(treat) %{y:.3f}<extra></extra>`,
      });
      traces.push({
        x: xs, y: xs.map((x) => POLICY.rule(p.on, p.off)(x, D)), type: "scatter", mode: "lines",
        line: { color: cur, width: 1.6, dash, shape: "hv" }, name: `your rule, ${label}`,
        hovertemplate: `your rule, ${label}<br>N/N(0) %{x:.2f}: %{y}<extra></extra>`,
      });
    }
    const layout = {
      ...baseLayout(), hovermode: "closest", margin: { l: 56, r: 12, t: 6, b: 44 },
      width: policyPlot.clientWidth, height: policyPlot.clientHeight,
      xaxis: axisStyle({ title: { text: "tumour size <i>N</i>/<i>N</i>(0)" }, range: [0, 1.3] }),
      yaxis: axisStyle({ title: { text: "probability of treating" }, range: [-0.05, 1.05] }),
      shapes: [{ type: "line", xref: "x", yref: "paper", x0: 1 + KAPPA, x1: 1 + KAPPA, y0: 0, y1: 1, line: { color: CB.cssVar("--ink"), width: 1.2, dash: "dash" } }],
    };
    Plotly.react(policyPlot, traces, layout, CB.PLOT_CONFIG);
  }

  function renderMetrics(blocks) {
    const months = (d) => CB.fmt(d / DAYS_PER_MONTH, 3);
    const cols = [
      ["progression (month)", (b) => (b.sim.metrics.ttp === null ? `none in ${months(HORIZON)}` : months(b.sim.metrics.ttp))],
      ["treated", (b) => `${Math.round(100 * b.sim.metrics.treatedFrac)}%`],
      ["holidays", (b) => b.sim.metrics.holidays],
      ["reward", (b) => b.sim.metrics.reward],
    ];
    let html = "<table class='metrics'><thead><tr><th>Strategy</th>" + cols.map((c) => `<th>${c[0]}</th>`).join("") + "</tr></thead><tbody>";
    for (const b of blocks) {
      html += `<tr><td><span class="dot" style="background:${b.colour}"></span>${CB.escapeHtml(b.short === b.name || b.params ? b.name : b.short)}</td>`;
      for (const [, f] of cols) {
        const v = f(b);
        html += typeof v === "number" ? `<td class="num">${CB.fmt(v, 4)}</td>` : `<td class="num">${v}</td>`;
      }
      html += "</tr>";
    }
    metricsEl.innerHTML = html + "</tbody></table>";
  }

  function renderDerived(p, rl) {
    const pr = presetById(p.preset);
    const rows = [
      ["agent", pr.source === "notebook checkpoint" ? "the notebook's agent" : `trained offline, seed ${pr.seed}`],
      ["decisions", `every ${AG.dt_days} days`],
      ["progression at", `N = ${CB.fmt(1 + KAPPA)} N(0)`],
    ];
    if (!isPreset(p)) rows.push(["", "edited patient: the agent was trained on the preset"]);
    derivedBox.innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><span class="v">${v}</span></div>`).join("");
  }

  // ======================= boot =======================

  function boot() {
    CB.renderTabs("rl");
    CB.renderPageMath(document.querySelector(".main"));
    buildSidebar();
    draw();
    window.addEventListener("themechange", scheduleDraw);
    CB.onWidthChange([mainPlot, policyPlot], scheduleDraw);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
