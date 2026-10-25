// Shared helpers for the visualiser pages: tab bar, controls, saved parameter
// sets, seeded random numbers, colours and polynomial roots.
// Plain script (no modules) so pages also work when opened from disk.

(function () {
  "use strict";

  const TABS = [
    { id: "pid", label: "Feedback (PID)", href: "pid.html" },
    { id: "oc", label: "Optimal control", href: "oc.html", soon: true },
    { id: "mpc", label: "Model predictive control", href: "mpc.html", soon: true },
    { id: "rl", label: "Reinforcement learning", href: "rl.html", soon: true },
  ];

  function renderTabs(active) {
    const nav = document.getElementById("tabs");
    if (!nav) return;
    nav.innerHTML = "";
    for (const t of TABS) {
      const el = document.createElement(t.soon ? "span" : "a");
      el.textContent = t.label;
      if (t.soon) {
        el.className = "soon";
        el.title = "Coming soon";
      } else {
        el.href = t.href;
        if (t.id === active) el.className = "active";
      }
      nav.appendChild(el);
    }
  }

  // ---------- maths ----------

  // mulberry32: small seeded PRNG, so saved runs with noise are reproducible
  function rng(seed) {
    let a = seed >>> 0;
    const uniform = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const normal = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = uniform();
      v = uniform();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return { uniform, normal };
  }

  // Roots of sum_i c[i] x^(n-i) (c[0] leading) by Durand–Kerner.
  // Returns [{re, im}]. Good enough for the low-order polynomials used here.
  function polyRoots(c) {
    while (c.length > 1 && Math.abs(c[0]) < 1e-14) c = c.slice(1);
    const n = c.length - 1;
    if (n < 1) return [];
    const a = c.map((x) => x / c[0]);
    const R = 1 + Math.max(...a.slice(1).map(Math.abs));
    let z = [];
    for (let k = 0; k < n; k++) {
      const th = (2 * Math.PI * k) / n + 0.4;
      z.push([R * 0.9 * Math.cos(th), R * 0.9 * Math.sin(th)]);
    }
    const mul = (p, q) => [p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]];
    const div = (p, q) => {
      const d = q[0] * q[0] + q[1] * q[1] || 1e-300;
      return [(p[0] * q[0] + p[1] * q[1]) / d, (p[1] * q[0] - p[0] * q[1]) / d];
    };
    for (let it = 0; it < 500; it++) {
      let delta = 0;
      const next = z.map((zi, i) => {
        let p = [1, 0];
        for (let k = 1; k <= n; k++) {
          const m = mul(p, zi);
          p = [m[0] + a[k], m[1]];
        }
        let q = [1, 0];
        z.forEach((zj, j) => { if (j !== i) q = mul(q, [zi[0] - zj[0], zi[1] - zj[1]]); });
        const step = div(p, q);
        delta = Math.max(delta, Math.hypot(step[0], step[1]));
        return [zi[0] - step[0], zi[1] - step[1]];
      });
      z = next;
      if (delta < 1e-13) break;
    }
    return z.map(([re, im]) => ({ re, im: Math.abs(im) < 1e-9 ? 0 : im }));
  }

  // ---------- formatting / colours ----------

  function fmt(x, digits = 3) {
    if (!isFinite(x)) return String(x);
    if (x === 0) return "0";
    const ax = Math.abs(x);
    if (ax >= 1e4 || ax < 1e-3) return x.toExponential(digits - 1);
    return Number(x.toPrecision(digits)).toString();
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // Random colour for a saved set, pushed away from hues already in use.
  function randomColour(used = []) {
    const usedHues = used.map((c) => c.h);
    let best = null, bestGap = -1;
    for (let i = 0; i < 24; i++) {
      const h = Math.floor(Math.random() * 360);
      const gap = usedHues.length
        ? Math.min(...usedHues.map((u) => Math.min(Math.abs(h - u), 360 - Math.abs(h - u))))
        : 360;
      if (gap > bestGap) { bestGap = gap; best = h; }
    }
    return { h: best, css: `hsl(${best} 68% 50%)` };
  }

  // ---------- storage (best effort; page works without it) ----------

  const store = {
    get(key, fallback) {
      try {
        const v = localStorage.getItem(key);
        return v === null ? fallback : JSON.parse(v);
      } catch (e) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore */ }
    },
  };

  // ---------- controls ----------

  function tex(el, src) {
    if (window.katex) katex.render(src, el, { throwOnError: false });
    else el.textContent = src;
  }

  // Slider + number box. spec: {key, label (TeX), min, max, step, hint, unit}
  // Returns {el, get, set}. onChange(key, value) fires on every edit.
  function slider(spec, value, onChange) {
    const wrap = document.createElement("div");
    wrap.className = "ctrl";
    const head = document.createElement("div");
    head.className = "ctrl-head";
    const lab = document.createElement("label");
    lab.className = "ctrl-label";
    tex(lab, spec.label + (spec.unit ? `\\;\\text{\\small(${spec.unit})}` : ""));
    const num = document.createElement("input");
    num.type = "number";
    num.step = "any";
    const range = document.createElement("input");
    range.type = "range";
    range.min = spec.min;
    range.max = spec.max;
    range.step = spec.step ?? (spec.max - spec.min) / 400;
    head.append(lab, num);
    wrap.append(head, range);
    if (spec.hint) {
      const h = document.createElement("div");
      h.className = "hint";
      h.textContent = spec.hint;
      wrap.append(h);
    }
    const set = (v) => {
      range.value = v;
      num.value = Number(Number(v).toPrecision(5));
    };
    set(value);
    range.addEventListener("input", () => {
      num.value = Number(Number(range.value).toPrecision(5));
      onChange(spec.key, parseFloat(range.value));
    });
    num.addEventListener("change", () => {
      const v = parseFloat(num.value);
      if (!isFinite(v)) return;
      range.value = v; // the box may go beyond the slider range on purpose
      onChange(spec.key, v);
    });
    return { el: wrap, set, get: () => parseFloat(num.value) };
  }

  function select(spec, value, onChange) {
    const wrap = document.createElement("div");
    wrap.className = "ctrl";
    const lab = document.createElement("div");
    lab.className = "ctrl-label";
    lab.textContent = spec.label;
    const sel = document.createElement("select");
    for (const [v, text] of spec.options) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = text;
      sel.appendChild(o);
    }
    sel.value = value;
    sel.addEventListener("change", () => onChange(spec.key, sel.value));
    wrap.append(lab, sel);
    return { el: wrap, set: (v) => (sel.value = v), get: () => sel.value };
  }

  function checkbox(spec, value, onChange) {
    const lab = document.createElement("label");
    lab.className = "ctrl-check";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !!value;
    box.addEventListener("change", () => onChange(spec.key, box.checked));
    const span = document.createElement("span");
    span.textContent = spec.label;
    lab.append(box, span);
    return { el: lab, set: (v) => (box.checked = !!v), get: () => box.checked };
  }

  function download(filename, text) {
    const blob = new Blob([text], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  window.CB = { renderTabs, rng, polyRoots, fmt, cssVar, randomColour, store, tex, slider, select, checkbox, download };
})();
