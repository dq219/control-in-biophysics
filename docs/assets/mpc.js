// Model-predictive-control visualiser: optogenetic control of GFP expression,
// following Sec. "Model predictive control" of the review and ModelPredictiveControl/GFP_expression.ipynb.
//   ẏ = A y + B [1, u]ᵀ,  y = [R, P, G] in fold changes from the dark steady state
//   y_{k+1} = A_k y_k + B_k u_k + η_k,   z_k = G_k + γ_k,   w_{k+1} = w_k + ρ_k,   w = [d_r, d_p, b_r, k_m]
// Every Δt = 10 min a particle filter updates its estimates of y and w from z, and the controller solves
// the finite-horizon LQ problem (Riccati recursion) with those estimates and applies the first control,
// clipped to [0, u_max]. Times in min, rates in 1/min.

(function () {
  "use strict";

  // ======================= model =======================

  const TS = 10; // sampling and control interval (min)
  const PARAMS = ["dr", "dp", "br", "km"];
  // notebook values, in the order of PARAMS: prior mean, random-walk step σ_ρ, prior width σ_θ
  const NB = {
    w: [0.0956, 0.0214, 0.0965, 0.0116],
    sigRho: [0.2539e-3, 0.0127e-3, 0.0037e-3, 0.2587e-3],
    sigTheta: [1.0155e-3, 0.0509e-3, 0.015e-3, 1.0347e-3],
  };

  const DEFAULTS = {
    shape: "sine", mean: 2.2, amp: 0.8, period: 300, T: 1200,
    H: 10, logCu: -2, umax: 4,
    sigEta: 0.01, sigGamma: 0.0025, lamRho: 1, lamTheta: 1, N: 2000, seed: 0,
    dr: NB.w[0], dp: NB.w[1], br: NB.w[2], km: NB.w[3],
  };
  const KEYS = Object.keys(DEFAULTS);

  function target(p, t) {
    const c = Math.cos((2 * Math.PI * t) / p.period);
    return p.mean - p.amp * (p.shape === "square" ? Math.sign(c) || 1 : c);
  }

  const newDisc = () => ({ A: new Float64Array(6), e: new Float64Array(3), b: new Float64Array(3) });

  // Exact discretisation over one interval (the notebook's `discretize`). A_k = e^{AΔt} is lower
  // triangular, stored as [a00, a10, a11, a20, a21, a22]; e and b are the constant and light columns of B_k.
  function discretize(dr, dp, br, km, D) {
    // rates are floored at 10⁻⁶ and kept apart, since the closed form divides by them and their differences
    const sep = (x, y) => (Math.abs(x - y) < 1e-8 ? y + 1e-8 : x);
    const a = Math.max(dr, 1e-6);
    const m = sep(Math.max(dp + km, 1e-6), a);
    const p = sep(sep(Math.max(dp, 1e-6), m), a);
    const ea = Math.exp(-a * TS), em = Math.exp(-m * TS), ep = Math.exp(-p * TS);
    const A10 = (m / (m - a)) * (ea - em);
    const A21 = (p / (p - m)) * (em - ep);
    const A20 = ((p * m) / (m - a)) * ((ea - ep) / (p - a) - (em - ep) / (p - m));
    D.A[0] = ea; D.A[1] = A10; D.A[2] = em; D.A[3] = A20; D.A[4] = A21; D.A[5] = ep;
    // first column of A⁻¹(A_k − I)
    const c0 = -(ea - 1) / a, c1 = c0 - A10 / m, c2 = c1 - A20 / p;
    D.e[0] = a * c0; D.e[1] = a * c1; D.e[2] = a * c2;
    D.b[0] = br * c0; D.b[1] = br * c1; D.b[2] = br * c2;
    return D;
  }

  // out[o..o+2] = A_k y + e + b u
  function step(D, y0, y1, y2, u, out, o) {
    const A = D.A, e = D.e, b = D.b;
    out[o] = A[0] * y0 + e[0] + b[0] * u;
    out[o + 1] = A[1] * y0 + A[2] * y1 + e[1] + b[1] * u;
    out[o + 2] = A[3] * y0 + A[4] * y1 + A[5] * y2 + e[2] + b[2] * u;
  }

  // ---------- LQ control over the horizon (review Eqs. MPC-SsBackPropRelationRecursion, MPC-optimalUMForm) ----------
  // 3×3 matrices are flat row-major arrays.

  const fullA = (D) => [D.A[0], 0, 0, D.A[1], D.A[2], 0, D.A[3], D.A[4], D.A[5]];
  const mv = (M, v) => [0, 1, 2].map((r) => M[3 * r] * v[0] + M[3 * r + 1] * v[1] + M[3 * r + 2] * v[2]);
  const mtv = (M, v) => [0, 1, 2].map((c) => M[c] * v[0] + M[3 + c] * v[1] + M[6 + c] * v[2]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  // Aᵀ X A
  function atxa(A, X) {
    const XA = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) XA.push(X[3 * r] * A[c] + X[3 * r + 1] * A[3 + c] + X[3 * r + 2] * A[6 + c]);
    const out = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out.push(A[r] * XA[c] + A[3 + r] * XA[3 + c] + A[6 + r] * XA[6 + c]);
    return out;
  }

  // Plan over the next H steps from the estimate y0, tracking G only (C_y = diag(0, 0, 1)).
  // targets are G* for y_1 … y_H. Returns the plan rolled forward on the model, with clipping;
  // its first control is the notebook's lqr_control.
  function controlPlan(y0, D, targets, Cu, umax) {
    const H = targets.length;
    const A = fullA(D), b = Array.from(D.b), e = Array.from(D.e);
    const S = new Array(H + 1), s = new Array(H + 1);
    S[H] = [0, 0, 0, 0, 0, 0, 0, 0, 1];
    s[H] = [0, 0, -targets[H - 1]];
    for (let i = H - 1; i >= 1; i--) {
      const Sb = mv(S[i + 1], b);
      const M = Cu + dot(b, Sb);
      // W = I − S b bᵀ / M, so W S = S − (S b)(S b)ᵀ / M and W v = v − S b (bᵀ v) / M
      const WS = S[i + 1].map((x, idx) => x - (Sb[(idx / 3) | 0] * Sb[idx % 3]) / M);
      S[i] = atxa(A, WS);
      S[i][8] += 1;
      const v = mv(S[i + 1], e).map((x, r) => x + s[i + 1][r]);
      const bv = dot(b, v) / M;
      s[i] = mtv(A, v.map((x, r) => x - Sb[r] * bv));
      s[i][2] -= targets[i - 1];
    }
    let x = Array.from(y0);
    const uPlan = [], gPlan = [];
    for (let i = 0; i < H; i++) {
      const Axe = mv(A, x).map((v, r) => v + e[r]);
      const M = Cu + dot(b, mv(S[i + 1], b));
      const raw = -dot(b, mv(S[i + 1], Axe).map((v, r) => v + s[i + 1][r])) / M;
      const u = Math.min(Math.max(raw, 0), umax);
      x = Axe.map((v, r) => v + b[r] * u);
      uPlan.push(u);
      gPlan.push(x[2]);
    }
    return { u: uPlan[0], uPlan, gPlan };
  }

  // ---------- particle filter (the notebook's ParticleFilter) ----------

  function makeFilter(p, sigRho, rng) {
    const N = Math.max(10, Math.round(p.N));
    const sigTh = NB.sigTheta.map((s) => s * p.lamTheta);
    const sg = Math.max(p.sigGamma, 1e-4); // a noiseless reading would give a degenerate likelihood
    let Y = new Float64Array(3 * N).fill(1), W = new Float64Array(4 * N);
    let Y2 = new Float64Array(3 * N), W2 = new Float64Array(4 * N);
    for (let n = 0; n < N; n++) for (let i = 0; i < 4; i++) W[4 * n + i] = NB.w[i] + rng.normal() * sigTh[i];
    const D = newDisc();
    const logw = new Float64Array(N), cum = new Float64Array(N);

    function update(u, z) {
      // (i) propagate: parameter random walk, then the state with noise
      let mx = -Infinity;
      for (let n = 0; n < N; n++) {
        const o = 4 * n;
        for (let i = 0; i < 4; i++) W2[o + i] = W[o + i] + rng.normal() * sigRho[i];
        discretize(W2[o], W2[o + 1], W2[o + 2], W2[o + 3], D);
        const q = 3 * n;
        step(D, Y[q], Y[q + 1], Y[q + 2], u, Y2, q);
        for (let i = 0; i < 3; i++) Y2[q + i] += rng.normal() * p.sigEta;
        const r = (Y2[q + 2] - z) / sg;
        logw[n] = -0.5 * r * r;
        if (logw[n] > mx) mx = logw[n];
      }
      // (ii) weights from the Gaussian likelihood of the reading
      let tot = 0;
      for (let n = 0; n < N; n++) { tot += Math.exp(logw[n] - mx); cum[n] = tot; }
      // (iii) multinomial resampling, using sorted uniforms built from exponential spacings
      let E = 0;
      const spac = new Float64Array(N + 1);
      for (let j = 0; j <= N; j++) { E += -Math.log(1 - rng.uniform()); spac[j] = E; }
      let n = 0;
      for (let j = 0; j < N; j++) {
        const uj = (spac[j] / E) * tot;
        while (n < N - 1 && cum[n] < uj) n++;
        for (let i = 0; i < 3; i++) Y[3 * j + i] = Y2[3 * n + i];
        for (let i = 0; i < 4; i++) W[4 * j + i] = W2[4 * n + i];
      }
    }

    const mean = (X, d) => {
      const out = new Array(d).fill(0);
      for (let n = 0; n < N; n++) for (let i = 0; i < d; i++) out[i] += X[d * n + i];
      return out.map((x) => x / N);
    };
    function thetaStd() {
      const mu = mean(W, 4), out = [0, 0, 0, 0];
      for (let n = 0; n < N; n++) for (let i = 0; i < 4; i++) out[i] += (W[4 * n + i] - mu[i]) ** 2;
      return out.map((x) => Math.sqrt(x / N));
    }
    return { update, stateMean: () => mean(Y, 3), thetaMean: () => mean(W, 4), thetaStd };
  }

  // independent random streams: 1 = the true system, 2 = the particle filter
  const seedFor = (seed, stream) => (Math.imul(Math.round(seed) + 1, 2654435761) ^ Math.imul(stream, 0x9e3779b9)) >>> 0;

  // Closed-loop run: the particle filter's estimates feed the controller. The true system has its own
  // random stream, so its noise realisation does not depend on the filter or on u.
  function simulate(p) {
    const tN = Math.round(p.T / TS) + 1;
    const H = Math.max(1, Math.round(p.H));
    const Cu = 10 ** p.logCu;
    const sigRho = NB.sigRho.map((s) => s * p.lamRho);
    const nrm = CB.rng(seedFor(p.seed, 1)).normal;
    const pf = makeFilter(p, sigRho, CB.rng(seedFor(p.seed, 2)));
    const D = newDisc();
    let w = PARAMS.map((k) => p[k]);
    let y = [1, 1, 1];
    const out = {
      t: [], G: [], z: [], target: [], U: [], Ghat: [], plans: [],
      wTrue: [[], [], [], []], wMean: [[], [], [], []], wStd: [[], [], [], []],
    };
    const record = (k, u, z) => {
      out.t.push(k * TS);
      out.G.push(y[2]);
      out.z.push(z);
      out.target.push(target(p, k * TS));
      out.U.push(u);
      out.Ghat.push(pf.stateMean()[2]);
      const wm = pf.thetaMean(), ws = pf.thetaStd();
      for (let i = 0; i < 4; i++) {
        out.wTrue[i].push(w[i]);
        out.wMean[i].push(wm[i]);
        out.wStd[i].push(ws[i]);
      }
    };
    record(0, 0, y[2]);
    for (let k = 1; k < tN; k++) {
      // decision at t_{k−1} for the step to t_k (plans[j] is the plan made at t_j)
      const future = Array.from({ length: H }, (_, i) => target(p, (k + i) * TS));
      const yhat = pf.stateMean();
      const what = pf.thetaMean();
      const plan = controlPlan(yhat, discretize(...what, D), future, Cu, p.umax);
      out.plans.push({ g0: yhat[2], u: plan.uPlan, g: plan.gPlan });
      // the true system: parameters drift, then the state steps with noise, then G is read
      w = w.map((x, i) => x + nrm() * sigRho[i]);
      const yn = [0, 0, 0];
      step(discretize(...w, D), y[0], y[1], y[2], plan.u, yn, 0);
      y = yn.map((x) => x + nrm() * p.sigEta);
      const z = y[2] + nrm() * p.sigGamma;
      pf.update(plan.u, z);
      record(k, plan.u, z);
    }
    out.metrics = metrics(out, p);
    return out;
  }

  function metrics(out, p) {
    let se = 0, mx = 0, su = 0, bound = 0;
    const n = out.t.length - 1;
    for (let k = 1; k <= n; k++) {
      const e = out.G[k] - out.target[k];
      se += e * e;
      mx = Math.max(mx, Math.abs(e));
      su += out.U[k];
      if (out.U[k] <= 1e-9 || out.U[k] >= p.umax - 1e-9) bound++;
    }
    const wErr = [0, 1, 2, 3].map((i) => Math.abs(out.wMean[i][n] - out.wTrue[i][n]) / Math.abs(out.wTrue[i][n]));
    return { rms: Math.sqrt(se / n), max: mx, meanU: su / n, bound: bound / n, wErr };
  }

  // runs are cached by their parameters, so scrubbing and redraws do not re-simulate
  const cache = new Map();
  function run(p) {
    const key = JSON.stringify(KEYS.map((k) => p[k]));
    if (!cache.has(key)) {
      if (cache.size > 40) cache.delete(cache.keys().next().value);
      cache.set(key, simulate(p));
    }
    return cache.get(key);
  }

  // ======================= sidebar specs =======================

  const SPEC = {
    target: [
      { key: "mean", label: "\\text{target mean}", min: 1, max: 4, step: 0.01, hint: "average target level of G (fold change)" },
      { key: "amp", label: "\\text{target amplitude}", min: 0, max: 2, step: 0.01, hint: "half the peak-to-peak swing of the target" },
      { key: "period", label: "\\text{target period}", unit: "min", min: 60, max: 600, step: 10 },
      { key: "T", label: "\\text{duration}", unit: "min", min: 300, max: 3000, step: 10, hint: "length of the experiment" },
    ],
    controller: [
      { key: "H", label: "\\text{prediction horizon (steps)}", min: 1, max: 40, step: 1, hint: "how many 10-min steps the controller plans ahead" },
      { key: "logCu", label: "\\log_{10}\\bm{C}_{u}", min: -5, max: 1, step: 0.05, hint: "weight on the light in the cost, relative to the tracking error" },
      { key: "umax", label: "\\text{maximum light}", min: 0.5, max: 10, step: 0.1, hint: "brightest light the LED can give; the light is kept between 0 and this" },
    ],
    noise: [
      { key: "sigEta", label: "\\text{state noise}", min: 0, max: 0.05, step: 0.0005, hint: "standard deviation of the noise added to R, P and G at each step" },
      { key: "sigGamma", label: "\\text{reading noise}", min: 0.0005, max: 0.05, step: 0.0005, hint: "standard deviation of the noise on each fluorescence reading" },
      { key: "lamRho", label: "\\text{parameter drift (×)}", min: 0, max: 20, step: 0.1, hint: "how fast the true parameters drift, as a multiple of the notebook's random-walk step" },
      { key: "lamTheta", label: "\\text{prior width (×)}", min: 0, max: 20, step: 0.1, hint: "spread of the filter's initial guess of the parameters, as a multiple of the notebook's" },
    ],
    filter: [
      { key: "N", label: "\\text{particles}", min: 100, max: 5000, step: 100 },
      { key: "seed", label: "\\text{random seed}", min: 0, max: 99, step: 1, hint: "for the noise and the filter" },
    ],
    truth: [
      { key: "dr", label: "d_{r}", unit: "1/min", min: 0.03, max: 0.2, step: 0.0005, hint: "mRNA degradation rate" },
      { key: "dp", label: "d_{p}", unit: "1/min", min: 0.005, max: 0.05, step: 0.0001, hint: "dilution rate of the proteins" },
      { key: "br", label: "b_{r}", unit: "1/min", min: 0.03, max: 0.2, step: 0.0005, hint: "transcription rate driven by the light" },
      { key: "km", label: "k_{m}", unit: "1/min", min: 0.002, max: 0.05, step: 0.0001, hint: "maturation rate of GFP" },
    ],
  };
  const PARAM_TEX = ["d_{r}", "d_{p}", "b_{r}", "k_{m}"];
  const PARAM_HTML = ["<i>d</i><sub>r</sub>", "<i>d</i><sub>p</sub>", "<i>b</i><sub>r</sub>", "<i>k</i><sub>m</sub>"];

  // ======================= state =======================

  const STORE_KEY = "cb.mpc.v1";
  const saved0 = CB.store.get(STORE_KEY, null);
  const state = {
    params: { ...DEFAULTS },
    scrub: 40,
    saved: [],
  };
  if (saved0) {
    state.params = { ...DEFAULTS, ...(saved0.params || {}) };
    state.saved = Array.isArray(saved0.saved) ? saved0.saved : [];
    if (Number.isFinite(saved0.scrub)) state.scrub = saved0.scrub;
  }
  readHash();

  function persist() {
    const { params, saved, scrub } = state;
    CB.store.set(STORE_KEY, { params, saved, scrub });
  }

  // share links: #shape=sine&H=10&...
  function readHash() {
    if (!location.hash || location.hash.length < 2) return;
    const q = new URLSearchParams(location.hash.slice(1));
    if (!q.has("shape")) return;
    const p = { ...DEFAULTS };
    for (const [k, v] of q.entries()) {
      if (k === "shape") p.shape = v === "square" ? "square" : "sine";
      else if (k in p && isFinite(parseFloat(v))) p[k] = parseFloat(v);
    }
    state.params = p;
  }

  function shareLink() {
    const q = new URLSearchParams();
    for (const k of KEYS) q.set(k, String(state.params[k]));
    return location.href.split("#")[0] + "#" + q.toString();
  }

  // ======================= sidebar =======================

  const sidebar = document.getElementById("sidebar");
  const { heading, button } = CB;
  let savedListEl = null;

  function onParam(key, value) {
    state.params[key] = value;
    scheduleDraw();
  }

  const sliders = (specs, p) => specs.map((s) => CB.slider(s, p[s.key], onParam).el);

  function note(text) {
    const el = document.createElement("p");
    el.className = "hint";
    el.textContent = text;
    return el;
  }

  function buildSidebar() {
    const p = state.params;
    sidebar.innerHTML = "";

    sidebar.append(heading("Target"));
    sidebar.append(CB.select({ key: "shape", label: "Shape", options: [["sine", "Sine wave (notebook)"], ["square", "Square wave"]] }, p.shape, onParam).el);
    sidebar.append(...sliders(SPEC.target, p));

    sidebar.append(heading("Controller"), ...sliders(SPEC.controller, p));

    sidebar.append(heading("Noise"), note("The filter assumes these same noise levels."), ...sliders(SPEC.noise, p));

    sidebar.append(heading("Particle filter"), ...sliders(SPEC.filter, p));

    const det = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = "True parameters at the start";
    sum.style.cursor = "pointer";
    sum.style.margin = "6px 0";
    det.append(sum, note("The filter's initial guess stays centred on the notebook values, so moving these tests how well it learns. Its estimates can only move as fast as its assumed drift allows, so a large offset also needs a larger parameter drift or prior width."), ...sliders(SPEC.truth, p));
    const rowT = document.createElement("div");
    rowT.className = "btn-row";
    rowT.append(button("Back to the notebook values", () => {
      PARAMS.forEach((k, i) => (p[k] = NB.w[i]));
      buildSidebar();
      scheduleDraw();
    }));
    det.append(rowT);
    det.open = PARAMS.some((k, i) => p[k] !== NB.w[i]);
    sidebar.append(det);

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
        state.params = { ...DEFAULTS };
        buildSidebar();
        scheduleDraw();
      })
    );
    sidebar.append(row2);
  }

  // ======================= saved sets =======================

  // hues of the current run and the plan, kept clear of saved colours
  const reservedHues = () => [{ h: 222 }, { h: 280 }];

  function summary(p) {
    return `horizon ${Math.round(p.H)} · Cu=${CB.fmt(10 ** p.logCu)} · state noise ${CB.fmt(p.sigEta)} · reading noise ${CB.fmt(p.sigGamma)} · drift ×${CB.fmt(p.lamRho)} · prior ×${CB.fmt(p.lamTheta)} · ${Math.round(p.N)} particles · seed ${Math.round(p.seed)}`;
  }

  function saveCurrent() {
    CB.addSavedSet(state.saved, state.params, reservedHues());
    scheduleDraw();
  }

  function renderSavedList() {
    CB.renderSavedList(savedListEl, state.saved, {
      onChange: scheduleDraw,
      onLoad: (s) => {
        state.params = { ...DEFAULTS, ...s.params };
        buildSidebar();
        scheduleDraw();
      },
      summary: (s) => summary({ ...DEFAULTS, ...s.params }),
      reserved: reservedHues(),
    });
  }

  function exportSets() {
    CB.download("mpc_saved_sets.json", JSON.stringify({ saved: state.saved }, null, 2));
  }

  function importSets() {
    CB.importJson((data) => {
      for (const s of data.saved || []) CB.addSavedSet(state.saved, { ...DEFAULTS, ...s.params }, reservedHues()).name = s.name || "Imported";
      scheduleDraw();
    });
  }

  // ======================= plots =======================

  const { PHI, axisStyle, baseLayout } = CB;
  const mainPlot = document.getElementById("plot-main");
  const legendEl = document.getElementById("legend");
  const metricsEl = document.getElementById("metrics");
  const gridEl = document.getElementById("param-grid");
  const paramLegendEl = document.getElementById("param-legend");
  const scrubEl = document.getElementById("scrub");
  const scrubT = document.getElementById("scrub-t");
  const playBtn = document.getElementById("play");
  let paramPlots = [];

  const scheduleDraw = CB.frameScheduler(() => draw());

  function buildParamGrid() {
    gridEl.innerHTML = "";
    paramPlots = PARAM_TEX.map((t) => {
      const panel = document.createElement("div");
      panel.className = "param-panel";
      const lab = document.createElement("div");
      lab.className = "param-label";
      CB.tex(lab, t + "\\;\\text{(1/min)}");
      const plot = document.createElement("div");
      plot.className = "param-plot";
      panel.append(lab, plot);
      gridEl.append(panel);
      return plot;
    });
  }

  function draw() {
    const p = state.params;
    const cur = CB.cssVar("--current");
    const horC = CB.cssVar("--horizon");
    const muted = CB.cssVar("--muted");

    // saved sets, then the current run
    const runs = [];
    for (const s of state.saved) {
      if (s.visible) runs.push({ name: s.name, colour: s.colour.css, width: 1.6, params: { ...DEFAULTS, ...s.params }, sim: run({ ...DEFAULTS, ...s.params }) });
    }
    const current = { name: "Current", colour: cur, width: 2.4, current: true, params: p, sim: run(p) };
    runs.push(current);

    renderSavedList();

    // receding-horizon scrubber: the plan made at t_j
    const sim = current.sim;
    const nPlans = sim.plans.length;
    state.scrub = Math.min(Math.max(Math.round(state.scrub), 0), nPlans - 1);
    scrubEl.max = nPlans - 1;
    scrubEl.value = state.scrub;
    const j = state.scrub;
    const tj = sim.t[j];
    const H = sim.plans[j].u.length;
    scrubT.textContent = `t = ${tj} min`;
    persist();

    // ----- G(t) and u(t) -----
    const traces = [];
    const shown = runs;
    const uSteps = (s) => [...s.U.slice(1), s.U[s.U.length - 1]]; // u_j held on [t_j, t_{j+1})
    traces.push({
      x: sim.t, y: sim.target, yaxis: "y", name: "Target", type: "scatter", mode: "lines",
      line: { color: muted, width: 1.6, dash: "dash", shape: p.shape === "square" ? "hv" : "linear" }, hovertemplate: "%{y:.3f}",
    });
    for (const r of shown) {
      traces.push({
        x: r.sim.t, y: r.sim.G, yaxis: "y", name: r.name, type: "scatter", mode: "lines",
        line: { color: r.colour, width: r.width, dash: r.dash || "solid" }, hovertemplate: "%{y:.3f}",
      });
      traces.push({
        x: r.sim.t, y: uSteps(r.sim), yaxis: "y2", name: r.name, type: "scatter", mode: "lines",
        line: { color: r.colour, width: r.width * 0.75, dash: r.dash || "solid", shape: "hv" }, hovertemplate: "%{y:.3f}",
        ...(r.current ? { fill: "tozeroy", fillcolor: CB.cssVar("--current-fill") } : {}),
      });
    }
    traces.push({
      x: sim.t, y: sim.z, yaxis: "y", name: "Reading", type: "scatter", mode: "markers",
      marker: { color: cur, size: 4.5, opacity: 0.6, line: { width: 0.6, color: CB.cssVar("--panel") } }, hovertemplate: "%{y:.3f}",
    });
    // the plan made at t_j over the next H steps
    const plan = sim.plans[j];
    const tPlan = Array.from({ length: H + 1 }, (_, i) => tj + i * TS);
    traces.push({
      x: tPlan, y: [plan.g0, ...plan.g], yaxis: "y", name: "Plan", type: "scatter", mode: "lines+markers",
      line: { color: horC, width: 2, dash: "dot" }, marker: { color: horC, size: 5 }, hovertemplate: "%{y:.3f}",
    });
    traces.push({
      x: tPlan, y: [...plan.u, plan.u[H - 1]], yaxis: "y2", name: "Plan", type: "scatter", mode: "lines+markers",
      line: { color: horC, width: 2, dash: "dot", shape: "hv" }, marker: { color: horC, size: 5 }, hovertemplate: "%{y:.3f}",
    });

    const W = mainPlot.clientWidth;
    const tEnd = tj + H * TS;
    const horizonShapes = (yref) => [
      { type: "rect", xref: "x", yref, x0: tj, x1: tEnd, y0: 0, y1: 1, fillcolor: horC, opacity: 0.08, line: { width: 0 }, layer: "below" },
      { type: "line", xref: "x", yref, x0: tj, x1: tj, y0: 0, y1: 1, line: { color: horC, width: 1.2 } },
    ];
    const uTop = Math.max(...shown.map((r) => r.params.umax)) * 1.08;
    const layout = {
      ...baseLayout(),
      width: W,
      height: Math.max(460, W / PHI), // golden-ratio panel
      margin: { l: 70, r: 16, t: 8, b: 44 },
      xaxis: axisStyle({ anchor: "y2", title: { text: "<i>t</i> (min)" }, range: [0, p.T] }),
      yaxis: axisStyle({ domain: [0.36, 1], title: { text: "GFP <i>G</i>(<i>t</i>)<br>(fold change)" } }),
      yaxis2: axisStyle({ domain: [0, 0.28], range: [0, uTop], title: { text: "Light <i>u</i>(<i>t</i>)" } }),
      shapes: [...horizonShapes("y domain"), ...horizonShapes("y2 domain")],
      annotations: [{
        // label the horizon, kept inside the plotted range
        x: (tj + Math.min(tEnd, p.T)) / 2, xref: "x", y: 1, yref: "y domain", yanchor: "top", showarrow: false,
        text: `horizon: ${H} steps (${H * TS} min)`, font: { color: horC }, bgcolor: CB.cssVar("--panel"), borderpad: 2,
      }],
    };
    Plotly.react(mainPlot, traces, layout, CB.PLOT_CONFIG);

    CB.renderLegend(legendEl, [
      ...runs.map((r) => [r.colour, r.name]),
      [cur, "Readings of G (current)", "markers"],
      [muted, "Target", "dash"],
      [horC, `Plan made at t = ${tj} min (current)`, "lines+markers"],
      [horC, `Prediction horizon, ${H} steps`, "shade"],
    ]);

    drawParams(runs, j);
    renderMetrics([...runs].reverse());
  }

  function drawParams(runs, j) {
    const cur = CB.cssVar("--current");
    const ink = CB.cssVar("--ink");
    const horC = CB.cssVar("--horizon");
    const current = runs[runs.length - 1];
    const sim = current.sim;
    const tj = sim.t[j];
    const mean = (r, i, width) => ({ x: r.sim.t, y: r.sim.wMean[i], type: "scatter", mode: "lines", name: r.name, line: { color: r.colour, width }, hovertemplate: "%{y:.4g}" });
    paramPlots.forEach((el, i) => {
      // mean ± 2σ of the current run: upper edge, then the lower edge filled to it
      const m = sim.wMean[i], sd = sim.wStd[i];
      const traces = [
        { x: sim.t, y: m.map((v, k) => v + 2 * sd[k]), type: "scatter", mode: "lines", line: { width: 0, color: cur }, hoverinfo: "skip" },
        {
          x: sim.t, y: m.map((v, k) => v - 2 * sd[k]), type: "scatter", mode: "lines", line: { width: 0, color: cur },
          fill: "tonexty", fillcolor: CB.cssVar("--current-fill"), hoverinfo: "skip",
        },
      ];
      for (const r of runs) if (!r.current) traces.push(mean(r, i, 1.2));
      traces.push({ x: sim.t, y: sim.wTrue[i], type: "scatter", mode: "lines", name: "True value", line: { color: ink, width: 1.4 }, hovertemplate: "%{y:.4g}" });
      traces.push(mean(current, i, 2));
      Plotly.react(el, traces, {
        ...baseLayout(),
        width: el.clientWidth,
        height: el.clientHeight,
        margin: { l: 52, r: 8, t: 4, b: 26 },
        xaxis: axisStyle({ range: [0, sim.t[sim.t.length - 1]], nticks: 4 }),
        yaxis: axisStyle({ nticks: 5, exponentformat: "power" }),
        shapes: [{ type: "line", xref: "x", yref: "paper", x0: tj, x1: tj, y0: 0, y1: 1, line: { color: horC, width: 1 } }],
      }, { ...CB.PLOT_CONFIG, displayModeBar: false });
    });
    CB.renderLegend(paramLegendEl, [
      [cur, "Current, particle-filter mean ± 2σ", "band"],
      ...runs.filter((r) => !r.current).map((r) => [r.colour, `${r.name}, particle-filter mean`]),
      [ink, "True value"],
      [horC, `Plan time, t = ${tj} min`],
    ]);
  }

  function renderMetrics(rows) {
    const pct = (x) => (x == null ? "–" : `${CB.fmt(100 * x, 2)}%`);
    const cols = [
      ["RMS error", (r) => r.sim.metrics.rms],
      ["max |error|", (r) => r.sim.metrics.max],
      ["mean <i>u</i>", (r) => r.sim.metrics.meanU],
      ["<i>u</i> at a bound", (r) => pct(r.sim.metrics.bound)],
      ...PARAM_HTML.map((h, i) => [`${h} error`, (r) => pct(r.sim.metrics.wErr?.[i])]),
    ];
    let html = "<table class='metrics'><thead><tr><th>Set</th>" + cols.map((c) => `<th>${c[0]}</th>`).join("") + "</tr></thead><tbody>";
    for (const r of rows) {
      html += `<tr><td><span class="dot" style="background:${r.colour}"></span>${CB.escapeHtml(r.name)}</td>`;
      for (const [, f] of cols) {
        const v = f(r);
        html += typeof v === "number" ? `<td class="num">${CB.fmt(v, 3)}</td>` : `<td class="num">${v}</td>`;
      }
      html += "</tr>";
    }
    metricsEl.innerHTML = html + "</tbody></table>";
  }

  // ======================= receding-horizon scrubber =======================

  let playTimer = null;
  function setPlaying(on) {
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = on ? "❚❚ Pause" : "▶ Play";
    if (!on) return;
    if (+scrubEl.value >= +scrubEl.max) state.scrub = 0;
    playTimer = setInterval(() => {
      if (state.scrub >= +scrubEl.max) return setPlaying(false);
      state.scrub++;
      scheduleDraw();
    }, 120);
  }

  // ======================= boot =======================

  function boot() {
    CB.renderTabs("mpc");
    CB.renderPageMath(document.querySelector(".main"));
    buildSidebar();
    buildParamGrid();
    scrubEl.addEventListener("input", () => {
      state.scrub = +scrubEl.value;
      scheduleDraw();
    });
    playBtn.addEventListener("click", () => setPlaying(!playTimer));
    draw();
    window.addEventListener("themechange", scheduleDraw);
    CB.onWidthChange([mainPlot, gridEl], scheduleDraw);
  }

  // model functions, for checking against the notebook from the console
  window.MPC = { discretize, newDisc, controlPlan, simulate, DEFAULTS };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
