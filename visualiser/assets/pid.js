// PID feedback visualiser. Two models, following Sec. "Feedback control" of the review:
//   toy     — linearised error dynamics  ε̇ = f_y ε − f_u u − η  with continuous-time PID
//   glucose — identified artificial-pancreas model (Ḡ(z), Δt = 5 min) with the
//             velocity-form discrete PID used in FeedbackControl/artificial_pancreas.ipynb
// Conventions: ε ≡ y* − y,  u = c_P ε + c_I ∫ε + c_D ε̇.

(function () {
  "use strict";

  // ======================= model definitions =======================

  const DT_GLU = 5; // min, sampling interval of the artificial pancreas
  const GLU = { a1: 2.48, a2: -2.0325, a3: 0.55125, b3: -0.25 };

  const DEFAULTS = {
    toy: {
      fy: -0.2, fu: 1.0,
      cP: 2.0, cI: 1.5, cD: 0.0,
      useP: true, useI: true, useD: true,
      dist: "pulse", A: 1.0, t0: 4.0, w: 1.8, k: 3.0,
      Gamma: 0.0, seed: 1, T: 20,
    },
    glucose: {
      mode: "tuned",
      K: -200, tau1: 247, tau2: 26, theta: 11, tauc: 40,
      cP: null, cI: null, cD: null, // manual gains; filled from tuning on first use
      useP: true, useI: true, useD: true,
      dist: "meal", A: 120, tauM: 40, t0h: 1, Ac: 0.13,
      clip: true, sigma: 0, beta: 0, sens: 1, seed: 1, hours: 24,
    },
  };

  function soptdGains(p) {
    const den = p.K * (p.tauc + p.theta);
    return { cP: (p.tau1 + p.tau2) / den, cI: 1 / den, cD: (p.tau1 * p.tau2) / den };
  }

  // gains actually used, after the mode choice and the P/I/D switches
  function effGains(model, p) {
    let g = model === "glucose" && p.mode === "tuned" ? soptdGains(p) : { cP: p.cP, cI: p.cI, cD: p.cD };
    return { cP: p.useP ? g.cP : 0, cI: p.useI ? g.cI : 0, cD: p.useD ? g.cD : 0 };
  }

  // ---------- toy model: RK4 on (ε, ∫ε), white noise by Euler–Maruyama ----------

  function toyEta(p, t) {
    if (p.dist === "pulse") return p.A * Math.exp(-0.5 * ((t - p.t0) / (p.w / 4)) ** 2);
    return p.A / (1 + Math.exp(-p.k * (t - p.t0)));
  }

  function simToy(p, g) {
    const dt = 0.01;
    const N = Math.round(p.T / dt);
    const stride = Math.max(1, Math.ceil(N / 1000));
    const a = 1 + p.fu * g.cD;
    const alpha = p.fy - p.fu * g.cP;
    if (!(a > 1e-9)) return { error: "1 + f_u c_D must be positive" };
    const rhs = (e, I, t) => [(alpha * e - p.fu * g.cI * I - toyEta(p, t)) / a, e];
    const r = CB.rng(p.seed);
    const noise = p.Gamma > 0;
    const out = { t: [], eta: [], e: [], u: [] };
    let e = 0, I = 0, binNoise = 0;
    let maxAbs = 0, ise = 0;
    const record = (t, etaNoise) => {
      const de = rhs(e, I, t)[0];
      out.t.push(t);
      out.eta.push(toyEta(p, t) + etaNoise);
      out.e.push(e);
      out.u.push(g.cP * e + g.cI * I + g.cD * de);
    };
    record(0, 0);
    for (let n = 0; n < N; n++) {
      const t = n * dt;
      const k1 = rhs(e, I, t);
      const k2 = rhs(e + 0.5 * dt * k1[0], I + 0.5 * dt * k1[1], t + 0.5 * dt);
      const k3 = rhs(e + 0.5 * dt * k2[0], I + 0.5 * dt * k2[1], t + 0.5 * dt);
      const k4 = rhs(e + dt * k3[0], I + dt * k3[1], t + dt);
      e += (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
      I += (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      if (noise) {
        const dW = Math.sqrt(p.Gamma * dt) * r.normal(); // ∫η dt over the step
        e -= dW / a;
        binNoise += dW;
      }
      maxAbs = Math.max(maxAbs, Math.abs(e));
      ise += e * e * dt;
      if ((n + 1) % stride === 0) {
        // noise is drawn as its average over the plotting bin
        record((n + 1) * dt, binNoise / (stride * dt));
        binNoise = 0;
      }
    }
    out.metrics = { maxAbs, ise, final: e };
    return out;
  }

  function toyPoles(p, g) {
    const a = 1 + p.fu * g.cD;
    const alpha = p.fy - p.fu * g.cP;
    // a λ² − α λ + f_u c_I = 0  (from (1+f_u c_D) ε̈ = (f_y − f_u c_P) ε̇ − f_u c_I ε)
    const closed = g.cI === 0 ? [{ re: alpha / a, im: 0 }, { re: 0, im: 0 }] : CB.polyRoots([a, -alpha, p.fu * g.cI]);
    return { closed, open: [{ re: p.fy, im: 0 }] };
  }

  function toyRegime(p, g, poles) {
    const a = 1 + p.fu * g.cD;
    const alpha = p.fy - p.fu * g.cP;
    const maxRe = Math.max(...poles.map((z) => z.re));
    if (maxRe > 1e-12) return "unstable";
    if (g.cI === 0) return "offset (no I)";
    const disc = alpha * alpha - 4 * a * p.fu * g.cI;
    const scale = alpha * alpha + 1e-12;
    if (Math.abs(disc) / scale < 1e-3) return "critical";
    return disc > 0 ? "overdamped" : "underdamped";
  }

  // ---------- glucose model: exactly the notebook's update order ----------

  function gluEta(p, tMin) {
    const meal = (t0, A) => {
      const tt = Math.max(tMin - t0, 0);
      return (A / p.tauM ** 2) * tt * Math.exp(-tt / p.tauM);
    };
    const t0 = p.t0h * 60;
    if (p.dist === "meal") return meal(t0, p.A);
    if (p.dist === "meals3") return meal(t0, p.A) + meal(t0 + 300, p.A) + meal(t0 + 660, p.A);
    return tMin >= t0 ? p.Ac * Math.tanh((tMin - t0) / 10) : 0; // constant offset
  }

  function simGlucose(p, g) {
    const N = Math.round((p.hours * 60) / DT_GLU);
    const b = GLU.b3 * p.sens;
    const r = CB.rng(p.seed);
    const x = [0], u = [0], eta = [0];
    const err = [0, 0];
    let uCache = 0, uIn = 0, dDprev = 0;
    for (let i = 0; i < N; i++) {
      const n = i + 1;
      const gIn = gluEta(p, i * DT_GLU);
      u.push(uIn);
      eta.push(gIn);
      const L = x.length;
      let xn = gIn + GLU.a1 * x[L - 1];
      if (n >= 2) xn += GLU.a2 * x[L - 2];
      if (n >= 3) xn += GLU.a3 * x[L - 3] + b * u[u.length - 3];
      x.push(xn);
      // controller (sees a noisy reading if σ > 0), set point y* = 0
      const yMeas = xn + (p.sigma > 0 ? p.sigma * r.normal() : 0);
      err.push(0 - yMeas);
      const E = err.length;
      const dP = g.cP * (err[E - 1] - err[E - 2]);
      const dI = g.cI * err[E - 1] * DT_GLU;
      const dDraw = (g.cD * (err[E - 1] - 2 * err[E - 2] + err[E - 3])) / DT_GLU;
      const dD = p.beta * dDprev + (1 - p.beta) * dDraw;
      dDprev = dD;
      let uNew = uCache + dP + dI + dD;
      if (p.clip) uNew = Math.max(0, uNew);
      uCache = uNew;
      uIn = uNew;
    }
    const t = x.map((_, k) => (k * DT_GLU) / 60);
    const uPer5 = u.map((v) => v / 12); // TDI/hr -> TDI per 5 min, as in the review's figure
    return {
      t, eta, e: x, u: uPer5,
      metrics: {
        peak: Math.max(...x),
        nadir: Math.min(...x),
        insulin: uPer5.reduce((s, v) => s + v, 0),
      },
    };
  }

  // Closed-loop poles of the linear loop (no clipping, no D filter):
  //   (1 − z⁻¹) A(z⁻¹) + β₃ z⁻³ Q(z⁻¹) = 0,
  //   A = 1 − α₁z⁻¹ − α₂z⁻² − α₃z⁻³,  Q = c_P(1 − z⁻¹) + c_I Δt + c_D(1 − z⁻¹)²/Δt.
  function gluPoles(p, g) {
    const pmul = (a, b) => {
      const c = new Array(a.length + b.length - 1).fill(0);
      a.forEach((ai, i) => b.forEach((bj, j) => (c[i + j] += ai * bj)));
      return c;
    };
    const padd = (a, b) => {
      const n = Math.max(a.length, b.length);
      return Array.from({ length: n }, (_, i) => (a[i] || 0) + (b[i] || 0));
    };
    const A = [1, -GLU.a1, -GLU.a2, -GLU.a3];
    const Q = padd(padd([g.cP, -g.cP], [g.cI * DT_GLU]), [g.cD / DT_GLU, (-2 * g.cD) / DT_GLU, g.cD / DT_GLU]);
    const charQ = padd(pmul([1, -1], A), pmul([0, 0, 0, GLU.b3 * p.sens], Q));
    // coefficients of z⁻ⁱ become coefficients of z^(deg − i): already leading-first
    const closed = CB.polyRoots(charQ);
    return { closed, open: [{ re: 0.98, im: 0 }, { re: 0.75, im: 0 }, { re: 0.75, im: 0 }] };
  }

  function run(model, p) {
    const g = effGains(model, p);
    const sim = model === "toy" ? simToy(p, g) : simGlucose(p, g);
    const poles = model === "toy" ? toyPoles(p, g) : gluPoles(p, g);
    return { g, sim, poles, params: p };
  }

  // ======================= sidebar specs =======================

  const SPEC = {
    toy: {
      plant: [
        { key: "fy", label: "f_y", min: -2, max: 1, step: 0.01, hint: "intrinsic relaxation, ∂f/∂y at y*" },
        { key: "fu", label: "f_u", min: 0.1, max: 3, step: 0.01, hint: "actuator gain, ∂f/∂u at y*" },
      ],
      gains: [
        { key: "cP", label: "c_{\\text{P}}", min: 0, max: 10, step: 0.01 },
        { key: "cI", label: "c_{\\text{I}}", min: 0, max: 10, step: 0.01 },
        { key: "cD", label: "c_{\\text{D}}", min: 0, max: 2, step: 0.005 },
      ],
      dist: {
        common: [
          { key: "A", label: "A", min: -3, max: 3, step: 0.01, hint: "disturbance amplitude" },
          { key: "t0", label: "t_0", min: 0, max: 60, step: 0.1 },
        ],
        pulse: [{ key: "w", label: "w", min: 0.1, max: 6, step: 0.05, hint: "pulse width" }],
        step: [{ key: "k", label: "k", min: 0.3, max: 20, step: 0.1, hint: "sigmoid steepness" }],
      },
      noise: [
        { key: "Gamma", label: "\\Gamma", min: 0, max: 0.5, step: 0.001, hint: "white-noise strength, ⟨η(t)η(t′)⟩ = Γδ(t − t′)" },
        { key: "seed", label: "\\text{seed}", min: 1, max: 100, step: 1 },
      ],
      sim: [{ key: "T", label: "T", min: 5, max: 60, step: 1, hint: "simulated time" }],
    },
    glucose: {
      tuning: [
        { key: "tauc", label: "\\tau_{\\text{c}}", unit: "min", min: 10, max: 200, step: 1, hint: "closed-loop time constant; typical 40–150 min" },
      ],
      soptd: [
        { key: "K", label: "K", unit: "mg/dL per TDI/hr", min: -400, max: -50, step: 1 },
        { key: "tau1", label: "\\tau_1", unit: "min", min: 20, max: 500, step: 1 },
        { key: "tau2", label: "\\tau_2", unit: "min", min: 0, max: 120, step: 1 },
        { key: "theta", label: "\\theta", unit: "min", min: 0, max: 60, step: 0.5 },
      ],
      gains: [
        { key: "cP", label: "c_{\\text{P}}", min: -0.12, max: 0, step: 0.0005 },
        { key: "cI", label: "c_{\\text{I}}", min: -5e-4, max: 0, step: 1e-6 },
        { key: "cD", label: "c_{\\text{D}}", min: -3, max: 0, step: 0.01 },
      ],
      dist: {
        meal: [
          { key: "A", label: "A", unit: "mg/dL", min: 0, max: 300, step: 1, hint: "total glucose of the meal" },
          { key: "tauM", label: "\\tau_{\\text{meal}}", unit: "min", min: 5, max: 120, step: 1 },
          { key: "t0h", label: "t_0", unit: "h", min: 0, max: 24, step: 0.25 },
        ],
        constant: [
          { key: "Ac", label: "A", unit: "mg/dL per step", min: 0, max: 0.5, step: 0.005, hint: "lost insulin sensitivity as a constant intake" },
          { key: "t0h", label: "t_0", unit: "h", min: 0, max: 24, step: 0.25 },
        ],
      },
      practical: [
        { key: "sens", label: "s", min: 0.25, max: 2.5, step: 0.01, hint: "insulin sensitivity: plant uses s·β₃ while the controller is unchanged" },
        { key: "sigma", label: "\\sigma", unit: "mg/dL", min: 0, max: 10, step: 0.1, hint: "sensor noise on the glucose reading" },
        { key: "beta", label: "\\beta", min: 0, max: 0.95, step: 0.01, hint: "derivative filter: ΔD_k ← βΔD_{k−1} + (1−β)ΔD_k" },
        { key: "seed", label: "\\text{seed}", min: 1, max: 100, step: 1 },
      ],
      sim: [{ key: "hours", label: "T", unit: "h", min: 6, max: 72, step: 1, hint: "simulated time" }],
    },
  };
  SPEC.glucose.dist.meals3 = SPEC.glucose.dist.meal;

  // ======================= state =======================

  const STORE_KEY = "cb.pid.v1";
  const saved0 = CB.store.get(STORE_KEY, null);
  const state = {
    model: "toy",
    params: { toy: { ...DEFAULTS.toy }, glucose: { ...DEFAULTS.glucose } },
    showOpen: true,
    saved: { toy: [], glucose: [] },
  };
  if (saved0) {
    state.model = saved0.model === "glucose" ? "glucose" : "toy";
    for (const m of ["toy", "glucose"]) {
      state.params[m] = { ...DEFAULTS[m], ...(saved0.params?.[m] || {}) };
      state.saved[m] = Array.isArray(saved0.saved?.[m]) ? saved0.saved[m] : [];
    }
    state.showOpen = saved0.showOpen !== false;
  }
  fillManualGains(state.params.glucose);
  readHash();

  function fillManualGains(p) {
    if (p.cP === null || p.cI === null || p.cD === null) Object.assign(p, soptdGains(p));
  }

  function persist() {
    CB.store.set(STORE_KEY, { model: state.model, params: state.params, saved: state.saved, showOpen: state.showOpen });
  }

  // share links: #model=toy&cP=2&...
  function readHash() {
    if (!location.hash || location.hash.length < 2) return;
    const q = new URLSearchParams(location.hash.slice(1));
    const m = q.get("model");
    if (m !== "toy" && m !== "glucose") return;
    state.model = m;
    const p = { ...DEFAULTS[m] };
    for (const [k, v] of q.entries()) {
      if (k === "model" || !(k in p)) continue;
      const d = DEFAULTS[m][k];
      if (typeof d === "boolean") p[k] = v === "1";
      else if (typeof d === "string") p[k] = v;
      else p[k] = parseFloat(v);
    }
    if (m === "glucose") fillManualGains(p);
    state.params[m] = p;
  }

  function shareLink() {
    const q = new URLSearchParams({ model: state.model });
    for (const [k, v] of Object.entries(state.params[state.model])) {
      q.set(k, typeof v === "boolean" ? (v ? "1" : "0") : String(v));
    }
    return location.href.split("#")[0] + "#" + q.toString();
  }

  // ======================= sidebar =======================

  const sidebar = document.getElementById("sidebar");

  function onParam(key, value) {
    state.params[state.model][key] = value;
    scheduleDraw();
  }

  function heading(text) {
    const h = document.createElement("h2");
    h.textContent = text;
    return h;
  }

  function segmented(options, value, onPick) {
    const div = document.createElement("div");
    div.className = "segmented";
    for (const [v, label] of options) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      if (v === value) b.className = "on";
      b.addEventListener("click", () => onPick(v));
      div.appendChild(b);
    }
    return div;
  }

  function sliders(specs, p) {
    return specs.map((s) => CB.slider(s, p[s.key], onParam).el);
  }

  function termSwitches(p) {
    const row = document.createElement("div");
    row.className = "btn-row";
    for (const [key, label] of [["useP", "P"], ["useI", "I"], ["useD", "D"]]) {
      const c = CB.checkbox({ key, label }, p[key], onParam);
      c.el.style.margin = "0 10px 0 0";
      row.appendChild(c.el);
    }
    return row;
  }

  let derivedBox = null; // live readout of tuned gains (glucose) or eigenvalues (toy)

  function buildSidebar() {
    const m = state.model;
    const p = state.params[m];
    const S = SPEC[m];
    sidebar.innerHTML = "";

    sidebar.append(heading("Model"));
    sidebar.append(
      segmented([["toy", "Toy model"], ["glucose", "Artificial pancreas"]], m, (v) => {
        state.model = v;
        buildSidebar();
        scheduleDraw();
      })
    );

    if (m === "toy") {
      sidebar.append(heading("Plant (linearised)"), ...sliders(S.plant, p));
      sidebar.append(heading("Controller"));
      sidebar.append(termSwitches(p), ...sliders(S.gains, p));
    } else {
      sidebar.append(heading("Controller"));
      sidebar.append(
        segmented([["tuned", "SOPTD tuning"], ["manual", "Manual gains"]], p.mode, (v) => {
          if (v === "manual" && p.mode === "tuned") Object.assign(p, soptdGains(p)); // start from the tuned gains
          p.mode = v;
          buildSidebar();
          scheduleDraw();
        })
      );
      sidebar.append(termSwitches(p));
      if (p.mode === "tuned") {
        sidebar.append(...sliders(S.tuning, p));
        const det = document.createElement("details");
        const sum = document.createElement("summary");
        sum.textContent = "SOPTD model  (K, τ₁, τ₂, θ)";
        sum.style.cursor = "pointer";
        sum.style.margin = "6px 0";
        det.append(sum, ...sliders(S.soptd, p));
        sidebar.append(det);
      } else {
        sidebar.append(...sliders(S.gains, p));
      }
    }
    derivedBox = document.createElement("div");
    derivedBox.className = "derived";
    sidebar.append(derivedBox);

    sidebar.append(heading("Disturbance η"));
    if (m === "toy") {
      sidebar.append(
        CB.select({ key: "dist", label: "Type", options: [["pulse", "Pulse (Gaussian)"], ["step", "Step (sigmoid)"]] }, p.dist, (k, v) => {
          p.dist = v;
          p.t0 = v === "pulse" ? 4 : 6; // the demo notebook's choices
          buildSidebar();
          scheduleDraw();
        }).el
      );
      sidebar.append(...sliders([...S.dist.common, ...S.dist[p.dist]], p), ...sliders(S.noise, p));
    } else {
      sidebar.append(
        CB.select(
          { key: "dist", label: "Type", options: [["meal", "Meal"], ["meals3", "Three meals (t₀, +5 h, +11 h)"], ["constant", "Constant (lost sensitivity)"]] },
          p.dist,
          (k, v) => { p.dist = v; buildSidebar(); scheduleDraw(); }
        ).el
      );
      sidebar.append(...sliders(S.dist[p.dist], p));
      sidebar.append(heading("Practical considerations"));
      sidebar.append(CB.checkbox({ key: "clip", label: "Insulin cannot be negative (u ≥ 0)" }, p.clip, onParam).el);
      sidebar.append(...sliders(S.practical, p));
    }

    sidebar.append(heading("Display"));
    sidebar.append(...sliders(S.sim, p));
    sidebar.append(
      CB.checkbox({ key: "showOpen", label: "Show uncontrolled response" }, state.showOpen, (k, v) => {
        state.showOpen = v;
        scheduleDraw();
      }).el
    );

    sidebar.append(heading("Saved parameter sets"));
    const row = document.createElement("div");
    row.className = "btn-row";
    row.append(
      button("Save current", saveCurrent, "primary"),
      button("Export", exportSets),
      button("Import", importSets),
      button("Clear", () => {
        if (state.saved[m].length && confirm("Delete all saved sets for this model?")) {
          state.saved[m] = [];
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
        state.params[m] = { ...DEFAULTS[m] };
        if (m === "glucose") fillManualGains(state.params[m]);
        buildSidebar();
        scheduleDraw();
      })
    );
    sidebar.append(row2);
  }

  function button(text, onClick, cls) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn" + (cls ? " " + cls : "");
    b.textContent = text;
    b.addEventListener("click", onClick);
    return b;
  }

  // ======================= saved sets =======================

  let savedListEl = null;

  function summary(model, p) {
    const g = effGains(model, p);
    const gains = `cP=${CB.fmt(g.cP)} cI=${CB.fmt(g.cI)} cD=${CB.fmt(g.cD)}`;
    if (model === "toy") return `${gains} · fy=${CB.fmt(p.fy)} · ${p.dist}`;
    const how = p.mode === "tuned" ? `τc=${p.tauc}` : "manual";
    return `${how} · ${gains} · ${p.dist}${p.sens !== 1 ? ` · s=${p.sens}` : ""}`;
  }

  function saveCurrent() {
    const m = state.model;
    const list = state.saved[m];
    const colour = CB.randomColour(list.map((s) => s.colour));
    list.push({
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      name: `Set ${list.length + 1}`,
      colour,
      visible: true,
      params: { ...state.params[m] },
    });
    scheduleDraw();
  }

  function renderSavedList() {
    const m = state.model;
    const list = state.saved[m];
    savedListEl.innerHTML = "";
    if (!list.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.style.display = "block";
      li.textContent = "Nothing saved yet. Saved sets stay on the plots in their own colour.";
      savedListEl.appendChild(li);
      return;
    }
    for (const s of list) {
      const li = document.createElement("li");
      if (!s.visible) li.className = "hidden-set";
      const vis = document.createElement("input");
      vis.type = "checkbox";
      vis.checked = s.visible;
      vis.title = "Show on plots";
      vis.addEventListener("change", () => { s.visible = vis.checked; scheduleDraw(); });
      const sw = document.createElement("button");
      sw.className = "swatch";
      sw.style.background = s.colour.css;
      sw.title = "New random colour";
      sw.addEventListener("click", () => {
        s.colour = CB.randomColour(list.filter((o) => o !== s).map((o) => o.colour));
        scheduleDraw();
      });
      const name = document.createElement("input");
      name.type = "text";
      name.value = s.name;
      name.addEventListener("change", () => { s.name = name.value || s.name; scheduleDraw(); });
      const acts = document.createElement("span");
      const load = document.createElement("button");
      load.className = "icon-btn";
      load.textContent = "load";
      load.title = "Copy these parameters into the current set";
      load.addEventListener("click", () => {
        state.params[m] = { ...DEFAULTS[m], ...s.params };
        buildSidebar();
        scheduleDraw();
      });
      const del = document.createElement("button");
      del.className = "icon-btn del";
      del.textContent = "✕";
      del.title = "Delete";
      del.addEventListener("click", () => {
        state.saved[m] = list.filter((o) => o !== s);
        scheduleDraw();
      });
      acts.append(load, del);
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = summary(m, s.params);
      li.append(vis, sw, name, acts, meta);
      savedListEl.appendChild(li);
    }
  }

  function exportSets() {
    CB.download(`pid_saved_sets_${state.model}.json`, JSON.stringify({ model: state.model, saved: state.saved[state.model] }, null, 2));
  }

  function importSets() {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "application/json,.json";
    inp.addEventListener("change", async () => {
      try {
        const data = JSON.parse(await inp.files[0].text());
        const m = data.model === "glucose" ? "glucose" : "toy";
        for (const s of data.saved || []) {
          state.saved[m].push({
            ...s,
            id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
            params: { ...DEFAULTS[m], ...s.params },
          });
        }
        state.model = m;
        buildSidebar();
        scheduleDraw();
      } catch (e) {
        alert("Could not read that file: " + e.message);
      }
    });
    inp.click();
  }

  // ======================= plots =======================

  const mainPlot = document.getElementById("plot-main");
  const polePlot = document.getElementById("plot-poles");
  const legendEl = document.getElementById("legend");
  const metricsEl = document.getElementById("metrics");

  let drawPending = false;
  function scheduleDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => {
      drawPending = false;
      draw();
    });
  }

  function axisStyle(extra) {
    return {
      gridcolor: CB.cssVar("--grid"),
      zerolinecolor: CB.cssVar("--rule"),
      linecolor: CB.cssVar("--rule"),
      tickfont: { size: 11 },
      automargin: true,
      ...extra,
    };
  }

  function baseLayout() {
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: CB.cssVar("--serif"), color: CB.cssVar("--ink"), size: 13 },
      showlegend: false,
      hovermode: "x unified",
      hoverlabel: { font: { family: CB.cssVar("--sans"), size: 11 } },
    };
  }

  function draw() {
    const m = state.model;
    const p = state.params[m];
    const ink = CB.cssVar("--current");
    const muted = CB.cssVar("--muted");

    const runs = [];
    for (const s of state.saved[m]) if (s.visible) runs.push({ name: s.name, colour: s.colour.css, width: 1.6, ...run(m, { ...DEFAULTS[m], ...s.params }) });
    runs.push({ name: "Current", colour: ink, width: 2.6, current: true, ...run(m, p) });
    const current = runs[runs.length - 1];

    document.body.dataset.model = m;
    renderDerived(m, p, current);
    renderSavedList();
    persist();

    // ----- time series -----
    const tUnit = m === "toy" ? "t" : "t (h)";
    const labels =
      m === "toy"
        ? ["Disturbance <i>η</i>(<i>t</i>)", "Error <i>ε</i>(<i>t</i>) = <i>y</i>* − <i>y</i>", "Control <i>u</i>(<i>t</i>)"]
        : ["Glucose intake <i>η<sub>k</sub></i><br>(mg/dL per 5 min)", "Blood glucose <i>y<sub>k</sub></i><br>(mg/dL from target)", "Insulin per 5 min<br>(TDI)"];
    const traces = [];
    const add = (r, key, axis, extra = {}) =>
      traces.push({
        x: r.sim.t, y: r.sim[key], xaxis: "x", yaxis: axis, name: r.name, type: "scatter", mode: "lines",
        line: { color: r.colour, width: r.width, ...(extra.line || {}) },
        hovertemplate: "%{y:.4g}", ...extra.trace,
      });

    if (state.showOpen) {
      const open = run(m, { ...p, useP: false, useI: false, useD: false });
      if (!open.sim.error) add({ ...open, name: "Uncontrolled", colour: muted, width: 1.3 }, "e", "y2", { line: { dash: "dot" } });
    }
    for (const r of runs) {
      if (r.sim.error) continue;
      add(r, "eta", "y");
      add(r, "e", "y2");
      add(r, "u", "y3");
    }
    const layout = {
      ...baseLayout(),
      margin: { l: 70, r: 16, t: 8, b: 44 },
      xaxis: axisStyle({ anchor: "y3", title: { text: tUnit }, ...(m === "glucose" ? { dtick: 6 } : {}) }),
      yaxis: axisStyle({ domain: [0.72, 1], title: { text: labels[0], font: { size: 12 } } }),
      yaxis2: axisStyle({ domain: [0.37, 0.67], title: { text: labels[1], font: { size: 12 } } }),
      yaxis3: axisStyle({ domain: [0, 0.32], title: { text: labels[2], font: { size: 12 } } }),
      shapes: [0.72, 0.37, 0].map((y0, i) => ({
        type: "line", xref: "paper", yref: ["y", "y2", "y3"][i], x0: 0, x1: 1, y0: 0, y1: 0,
        line: { color: CB.cssVar("--rule"), width: 1 },
      })),
    };
    Plotly.react(mainPlot, traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] });

    // legend
    legendEl.innerHTML = "";
    const item = (colour, text, dashed) => {
      const s = document.createElement("span");
      const i = document.createElement("i");
      i.style.background = dashed ? `repeating-linear-gradient(90deg, ${colour} 0 3px, transparent 3px 6px)` : colour;
      s.append(i, document.createTextNode(text));
      legendEl.appendChild(s);
    };
    for (const r of runs) item(r.colour, r.name);
    if (state.showOpen) item(muted, "Uncontrolled", true);
    const err = current.sim.error;
    document.getElementById("sim-error").textContent = err ? "Cannot simulate the current set: " + err : "";

    drawPoles(m, runs);
    renderMetrics(m, runs);
  }

  function drawPoles(m, runs) {
    const muted = CB.cssVar("--muted");
    const traces = [];
    const cur = runs[runs.length - 1];
    traces.push({
      x: cur.poles.open.map((z) => z.re), y: cur.poles.open.map((z) => z.im),
      mode: "markers", type: "scatter", name: "Open loop",
      marker: { symbol: "x-thin-open", size: 11, color: muted, line: { width: 2, color: muted } },
      hovertemplate: "open loop<br>%{x:.4g} %{y:+.4g}i<extra></extra>",
    });
    for (const r of runs) {
      traces.push({
        x: r.poles.closed.map((z) => z.re), y: r.poles.closed.map((z) => z.im),
        mode: "markers", type: "scatter", name: r.name,
        marker: { size: r.current ? 10 : 8, color: r.colour, line: { width: 1, color: CB.cssVar("--panel") } },
        hovertemplate: `${r.name}<br>%{x:.4g} %{y:+.4g}i<extra></extra>`,
      });
    }
    const layout = { ...baseLayout(), hovermode: "closest", margin: { l: 56, r: 12, t: 6, b: 44 } };
    if (m === "toy") {
      // explicit ranges, always showing the imaginary axis and some of the unstable side
      const all = runs.flatMap((r) => r.poles.closed).concat(cur.poles.open);
      const re = all.map((z) => z.re), im = all.map((z) => Math.abs(z.im));
      const lo = Math.min(...re, 0), hi = Math.max(...re, 0);
      const pad = 0.15 * (hi - lo || 1);
      const xr = [lo - pad, hi + Math.max(pad, 0.25 * (hi - lo || 1))];
      const ym = Math.max(...im, 0.25 * (xr[1] - xr[0])) * 1.2;
      layout.xaxis = axisStyle({ title: { text: "Re <i>λ</i>" }, range: xr });
      layout.yaxis = axisStyle({ title: { text: "Im <i>λ</i>" }, range: [-ym, ym] });
      layout.shapes = [{
        type: "rect", xref: "x", yref: "paper", x0: 0, x1: xr[1], y0: 0, y1: 1,
        fillcolor: CB.cssVar("--danger"), opacity: 0.07, line: { width: 0 }, layer: "below",
      }];
    } else {
      layout.xaxis = axisStyle({ title: { text: "Re <i>z</i>" }, scaleanchor: "y", constrain: "domain" });
      layout.yaxis = axisStyle({ title: { text: "Im <i>z</i>" } });
      layout.shapes = [{
        type: "circle", xref: "x", yref: "y", x0: -1, x1: 1, y0: -1, y1: 1,
        line: { color: CB.cssVar("--muted"), width: 1, dash: "dot" },
      }];
    }
    Plotly.react(polePlot, traces, layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] });
  }

  const cstr = (z) => (z.im === 0 ? CB.fmt(z.re) : `${CB.fmt(z.re)} ${z.im < 0 ? "−" : "+"} ${CB.fmt(Math.abs(z.im))}i`);

  function renderMetrics(m, runs) {
    const cols =
      m === "toy"
        ? [["max |ε|", (r) => r.sim.metrics.maxAbs], ["∫ε² dt", (r) => r.sim.metrics.ise], ["ε(T)", (r) => r.sim.metrics.final],
           ["λ±", (r) => uniq(r.poles.closed).map(cstr).join(", ")], ["regime", (r) => toyRegime(r.params, r.g, r.poles.closed)]]
        : [["peak y (mg/dL)", (r) => r.sim.metrics.peak], ["nadir y (mg/dL)", (r) => r.sim.metrics.nadir],
           ["insulin (TDI)", (r) => r.sim.metrics.insulin], ["max |z|", (r) => Math.max(...r.poles.closed.map((z) => Math.hypot(z.re, z.im)))]];
    let html = "<table class='metrics'><thead><tr><th>Set</th>" + cols.map((c) => `<th>${c[0]}</th>`).join("") + "</tr></thead><tbody>";
    for (const r of [...runs].reverse()) {
      if (r.sim.error) continue;
      html += `<tr><td><span class="dot" style="background:${r.colour}"></span>${escapeHtml(r.name)}</td>`;
      for (const [, f] of cols) {
        const v = f(r);
        html += typeof v === "number" ? `<td class="num">${CB.fmt(v)}</td>` : `<td>${v}</td>`;
      }
      html += "</tr>";
    }
    metricsEl.innerHTML = html + "</tbody></table>";
  }

  function uniq(zs) {
    const out = [];
    for (const z of zs) if (!out.some((o) => Math.abs(o.re - z.re) < 1e-9 && Math.abs(o.im + z.im) < 1e-9 && z.im !== 0)) out.push(z);
    return out;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function renderDerived(m, p, current) {
    if (!derivedBox) return;
    const g = current.g;
    const rows = [];
    if (m === "glucose") {
      rows.push(["c_{\\text{P}}", CB.fmt(g.cP, 4)], ["c_{\\text{I}}", CB.fmt(g.cI, 4)], ["c_{\\text{D}}", CB.fmt(g.cD, 4)]);
      const maxZ = Math.max(...current.poles.closed.map((z) => Math.hypot(z.re, z.im)));
      const verdict = Math.abs(maxZ - 1) < 1e-6 ? " marginal" : maxZ < 1 ? " stable" : " unstable";
      rows.push(["\\max|z|\\ \\text{(linear loop)}", CB.fmt(maxZ, 4) + verdict]);
    } else {
      rows.push(["f_y-f_u c_{\\text{P}}", CB.fmt(p.fy - p.fu * g.cP)]);
      rows.push(["1+f_u c_{\\text{D}}", CB.fmt(1 + p.fu * g.cD)]);
      const a = 1 + p.fu * g.cD, alpha = p.fy - p.fu * g.cP;
      if (p.fu > 0 && a > 0) rows.push(["\\text{critical } c_{\\text{I}}", CB.fmt((alpha * alpha) / (4 * a * p.fu))]);
      rows.push(["\\text{regime}", toyRegime(p, g, current.poles.closed)]);
    }
    const texOf = (k) => (window.katex ? katex.renderToString(k, { throwOnError: false }) : k);
    derivedBox.innerHTML = rows.map(([k, v]) => `<div><span>${texOf(k)}</span><span class="v">${v}</span></div>`).join("");
  }

  // ======================= theory visibility + boot =======================

  function boot() {
    CB.renderTabs("pid");
    if (window.renderMathInElement) {
      renderMathInElement(document.querySelector(".main"), {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
        ],
        macros: { "\\dd": "\\mathrm{d}" },
        throwOnError: false,
      });
    }
    buildSidebar();
    draw();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", scheduleDraw);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
