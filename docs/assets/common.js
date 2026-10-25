// Shared helpers for the visualiser pages: tab bar, controls, saved parameter
// sets, seeded random numbers, colours and polynomial roots.
// Plain script (no modules) so pages also work when opened from disk.

(function () {
  "use strict";

  const TABS = [
    { id: "pid", label: "Feedback (PID)", href: "pid.html" },
    { id: "oc", label: "Optimal control", href: "oc.html" },
    { id: "mpc", label: "Model predictive control", href: "mpc.html", soon: true },
    { id: "rl", label: "Reinforcement learning", href: "rl.html" },
  ];

  function renderTabs(active) {
    const nav = document.getElementById("tabs");
    if (!nav) return;
    nav.innerHTML = "";
    renderThemeToggle();
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

  // ---------- light / dark theme ----------
  // The saved choice is applied by a one-line script in each page's <head> (no flash);
  // this adds the toggle button and fires "themechange" so pages can redraw plots.

  const THEME_KEY = "cb.theme";

  function renderThemeToggle() {
    const nav = document.getElementById("tabs");
    if (!nav || nav.parentElement.classList.contains("header-right")) return;
    const wrap = document.createElement("div");
    wrap.className = "header-right";
    nav.replaceWith(wrap);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    const label = () => {
      const dark = document.documentElement.dataset.theme === "dark";
      btn.textContent = dark ? "☀ Light" : "☾ Dark";
      btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
    };
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
      label();
      window.dispatchEvent(new Event("themechange"));
    });
    label();
    wrap.append(nav, btn);
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
    if (ax >= 1e4 || ax < 1e-3) {
      // m × 10ⁿ with a superscript exponent
      const [m, e] = x.toExponential(digits - 1).split("e");
      const sup = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
      return `${Number(m)}×10${String(parseInt(e, 10)).replace(/./g, (c) => sup[c])}`;
    }
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
    tex(lab, spec.label + (spec.unit ? `\\;\\text{(${spec.unit})}` : ""));
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

  // ---------- small sidebar pieces ----------

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

  function button(text, onClick, cls) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn" + (cls ? " " + cls : "");
    b.textContent = text;
    b.addEventListener("click", onClick);
    return b;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  // ---------- saved parameter sets ----------
  // A set is {id, name, colour, visible, params}. Lists are edited in place.

  const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 6);

  // reserved: [{h}] hues kept clear of, e.g. the current run's colour
  function addSavedSet(list, params, reserved = []) {
    const colour = randomColour([...list.map((s) => s.colour), ...reserved]);
    const set = { id: newId(), name: `Set ${list.length + 1}`, colour, visible: true, params: { ...params } };
    list.push(set);
    return set;
  }

  // opts: {onChange(), onLoad(set), summary(set) -> string, reserved: [{h}]}
  function renderSavedList(ul, list, opts) {
    ul.innerHTML = "";
    if (!list.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.style.display = "block";
      li.textContent = "Nothing saved yet. Saved sets stay on the plots in their own colour.";
      ul.appendChild(li);
      return;
    }
    for (const s of list) {
      const li = document.createElement("li");
      if (!s.visible) li.className = "hidden-set";
      const vis = document.createElement("input");
      vis.type = "checkbox";
      vis.checked = s.visible;
      vis.title = "Show on plots";
      vis.addEventListener("change", () => { s.visible = vis.checked; opts.onChange(); });
      const sw = document.createElement("button");
      sw.className = "swatch";
      sw.style.background = s.colour.css;
      sw.title = "New random colour";
      sw.addEventListener("click", () => {
        s.colour = randomColour([...list.filter((o) => o !== s).map((o) => o.colour), ...(opts.reserved || [])]);
        opts.onChange();
      });
      const name = document.createElement("input");
      name.type = "text";
      name.value = s.name;
      name.addEventListener("change", () => { s.name = name.value || s.name; opts.onChange(); });
      const acts = document.createElement("span");
      const load = document.createElement("button");
      load.className = "icon-btn";
      load.textContent = "load";
      load.title = "Copy these parameters into the current set";
      load.addEventListener("click", () => opts.onLoad(s));
      const del = document.createElement("button");
      del.className = "icon-btn del";
      del.textContent = "✕";
      del.title = "Delete";
      del.addEventListener("click", () => {
        list.splice(list.indexOf(s), 1);
        opts.onChange();
      });
      acts.append(load, del);
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = opts.summary(s);
      li.append(vis, sw, name, acts, meta);
      ul.appendChild(li);
    }
  }

  // Ask for a JSON file and hand its parsed contents to onData.
  function importJson(onData) {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "application/json,.json";
    inp.addEventListener("change", async () => {
      try {
        onData(JSON.parse(await inp.files[0].text()));
      } catch (e) {
        alert("Could not read that file: " + e.message);
      }
    });
    inp.click();
  }

  // ---------- plot styling (Plotly) ----------

  const PHI = (1 + Math.sqrt(5)) / 2;

  function axisStyle(extra) {
    return {
      gridcolor: cssVar("--grid"),
      zerolinecolor: cssVar("--rule"),
      linecolor: cssVar("--rule"),
      automargin: true,
      ...extra,
    };
  }

  function baseLayout() {
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: cssVar("--serif"), color: cssVar("--ink"), size: parseFloat(cssVar("--fs")) },
      showlegend: false,
      hovermode: "x unified",
      hoverlabel: { font: { family: cssVar("--sans"), size: parseFloat(cssVar("--fs")) } },
    };
  }

  const PLOT_CONFIG = { displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] };

  // Legend above a plot: items are [colour, text, dashed?]
  function renderLegend(el, items) {
    el.innerHTML = "";
    for (const [colour, text, dashed] of items) {
      const s = document.createElement("span");
      const i = document.createElement("i");
      i.style.background = dashed ? `repeating-linear-gradient(90deg, ${colour} 0 3px, transparent 3px 6px)` : colour;
      s.append(i, document.createTextNode(text));
      el.appendChild(s);
    }
  }

  // Coalesce redraws into one animation frame.
  function frameScheduler(fn) {
    let pending = false;
    return () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => { pending = false; fn(); });
    };
  }

  // Redraw when any observed element changes width (plots are sized from their container).
  function onWidthChange(els, fn) {
    const lastW = new Map();
    const ro = new ResizeObserver((entries) => {
      for (const en of entries) {
        const w = Math.round(en.contentRect.width);
        if (lastW.get(en.target) !== w) { lastW.set(en.target, w); fn(); }
      }
    });
    els.forEach((el) => ro.observe(el));
  }

  // Typeset $…$ in the page body with the review's macros.
  function renderPageMath(root) {
    if (!window.renderMathInElement) return;
    renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      macros: { "\\dd": "\\mathrm{d}" },
      throwOnError: false,
    });
  }

  window.CB = {
    renderTabs, rng, polyRoots, fmt, cssVar, randomColour, store, tex, slider, select, checkbox, download,
    heading, segmented, button, escapeHtml, addSavedSet, renderSavedList, importJson,
    PHI, axisStyle, baseLayout, PLOT_CONFIG, renderLegend, frameScheduler, onWidthChange, renderPageMath,
  };
})();
