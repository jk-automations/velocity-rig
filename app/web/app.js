"use strict";
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else if (k.startsWith("on")) n[k] = v;
    else if (k === "style") Object.assign(n.style, v); else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  n.append(...kids.flat().filter(x => x != null && x !== false));
  return n;
};
const api = {
  async get(u) { const r = await fetch(u); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); },
  async post(u, body) { const r = await fetch(u, { method: "POST", body }); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); },
};
function toast(msg, bad) { const t = el("div", { class: "toast" + (bad ? " bad" : ""), text: msg }); $("#toasts").append(t); setTimeout(() => t.remove(), 5000); }
function lightbox(url) {
  const lb = $("#lightbox");
  lb.replaceChildren(/\.(mp4|webm|mov)$/i.test(url) ? el("video", { src: url, controls: true, autoplay: true, loop: true }) : el("img", { src: url }));
  lb.hidden = false; lb.onclick = e => { if (e.target === lb) { lb.hidden = true; lb.replaceChildren(); } };
}
const isVid = u => /\.(mp4|webm|mov)$/i.test(u);
const runUrl = p => "/runs/" + p.split("/").map(encodeURIComponent).join("/");
const vurl = u => runUrl(u) + "#t=0.1";
let CFG, LIB, FORM = {}, FILES = {}, cleanup = () => {};

/* ---------- ComfyUI graph diagram (ported from Open Adfield) ---------- */
const SVGNS = "http://www.w3.org/2000/svg";
function sv(tag, attrs, text) { const n = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); if (text != null) n.textContent = text; return n; }
const NODE_COLORS = [[/Loader|LoadImage|LoadVideo|LoRA/i, "#5878a0"], [/CLIPTextEncode|Conditioning|TextEncode|Prompt/i, "#6f8f58"],
  [/Sampler|Scheduler|Sigmas|Guider|Noise|CFG/i, "#b98a3e"], [/VAE|Latent|Upscale/i, "#8368a8"], [/Save|Preview|Output|Report|Compare/i, "#a65d5d"], [/Image|Video|Audio|Components/i, "#4d9696"]];
const nodeColor = ct => (CFG.node_classes || []).some(re => new RegExp(re).test(ct)) ? "var(--accent)" : (NODE_COLORS.find(([re]) => re.test(ct)) || [0, "#6b7078"])[1];
const KEYS = ["unet_name", "clip_name", "vae_name", "lora_name", "model_name", "preset", "strength", "min_fidelity", "text", "prompt", "sampler_name", "steps", "cfg", "seed", "noise_seed", "width", "height", "length", "frame_rate", "fps", "filename_prefix", "image", "file"];
function modelLabel(v) { const l = v.toLowerCase(); const m = (CFG.models || []).find(m => (m.match || []).some(p => l.includes(p))); return m ? m.name : null; }
function nodeLines(inputs) {
  const sc = Object.entries(inputs || {}).filter(([, v]) => !Array.isArray(v) && v !== null && typeof v !== "object");
  sc.sort((a, b) => { const ia = KEYS.indexOf(a[0]), ib = KEYS.indexOf(b[0]); return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib); });
  return sc.slice(0, 3).map(([k, v]) => { let t = (typeof v === "string" && modelLabel(v)) || String(v).replace(/\s+/g, " "); if (t.length > 24) t = t.slice(0, 23) + "..."; return `${k.slice(0, 12)}: ${t}`; });
}
function layoutGraph(graph) {
  const ids = Object.keys(graph), memo = {};
  const srcs = id => Object.values(graph[id].inputs || {}).filter(v => Array.isArray(v) && graph[String(v[0])]).map(v => String(v[0]));
  const depth = (id, seen = new Set()) => { if (memo[id] != null) return memo[id]; if (seen.has(id)) return 0; seen.add(id); return (memo[id] = srcs(id).reduce((m, s) => Math.max(m, depth(s, seen) + 1), 0)); };
  ids.forEach(i => depth(i));
  const cols = {}; ids.forEach(i => (cols[memo[i]] = cols[memo[i]] || []).push(i));
  const W = 214, HD = 28, LH = 15, GX = 74, GY = 16, pos = {}, maxD = Math.max(...Object.keys(cols).map(Number), 0); let H = 0;
  for (let d = 0; d <= maxD; d++) {
    const list = cols[d] || [];
    list.sort((a, b) => { const avg = id => { const s = srcs(id); return s.length ? s.reduce((t, x) => t + (pos[x] ? pos[x].y : 0), 0) / s.length : 0; }; return avg(a) - avg(b); });
    let y = 20;
    list.forEach(id => { const lines = nodeLines(graph[id].inputs), h = HD + 8 + lines.length * LH + 6; pos[id] = { x: 20 + d * (W + GX), y, w: W, h, lines }; y += h + GY; });
    H = Math.max(H, y);
  }
  const edges = [];
  ids.forEach(id => { let k = 0; Object.entries(graph[id].inputs || {}).forEach(([name, v]) => { if (Array.isArray(v) && pos[String(v[0])]) edges.push({ from: String(v[0]), to: id, name, k: k++ }); }); });
  return { pos, edges, W: 40 + (maxD + 1) * (W + GX) - GX, H: H + 10 };
}
function drawGraph(graph, onSelect) {
  const L = layoutGraph(graph), svg = sv("svg", { width: L.W, height: L.H, viewBox: `0 0 ${L.W} ${L.H}` }), eg = sv("g"), ng = sv("g"), edgeEls = [], nodeEls = {};
  svg.append(eg, ng);
  L.edges.forEach(e => {
    const a = L.pos[e.from], b = L.pos[e.to], x1 = a.x + a.w, y1 = a.y + 14, x2 = b.x, y2 = b.y + 14 + Math.min(e.k, 6) * 6;
    const p = sv("path", { class: "gedge", d: `M${x1} ${y1} C${x1 + 44} ${y1} ${x2 - 44} ${y2} ${x2} ${y2}` });
    p.append(sv("title", {}, `${e.from} -> ${e.to}.${e.name}`)); eg.append(p); edgeEls.push([e, p]);
  });
  Object.entries(L.pos).forEach(([id, p]) => {
    const n = graph[id], g = sv("g", { class: "gnode", transform: `translate(${p.x},${p.y})`, style: "cursor:pointer" });
    g.append(sv("rect", { class: "b", width: p.w, height: p.h, rx: 8 }), sv("rect", { width: 4, height: p.h - 12, x: 0, y: 6, rx: 2, fill: nodeColor(n.class_type) }),
      sv("text", { class: "hd", x: 14, y: 18 }, (n._meta && n._meta.title) || n.class_type), sv("text", { class: "stp", x: p.w - 8, y: 18, "text-anchor": "end" }, ""));
    p.lines.forEach((t, i) => g.append(sv("text", { class: "ln", x: 14, y: 44 + i * 15 }, t)));
    g.append(sv("title", {}, `${n.class_type} (${id})`)); g.onclick = () => onSelect && onSelect(id); ng.append(g); nodeEls[id] = g;
  });
  return { svg, W: L.W, H: L.H, pos: L.pos,
    select(id) { Object.entries(nodeEls).forEach(([k, g]) => g.classList.toggle("sel", k === id)); edgeEls.forEach(([e, p]) => p.classList.toggle("hl", e.from === id || e.to === id)); },
    live(id, label) { Object.entries(nodeEls).forEach(([k, g]) => { const on = k === id; g.classList.toggle("exec", on); g.querySelector(".stp").textContent = on ? (label || "") : ""; }); return id && L.pos[id] ? L.pos[id] : null; } };
}

function graphPanel(run) {
  let graphs = [], sel = null, d = null, rec = null, liveFile = null;
  const tabs = el("div", { class: "tabs" }), view = el("div", { class: "gview" }), bar = el("div", { class: "gbar" }), detail = el("div", { class: "nd-detail", hidden: true });
  const root = el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("div", {}, el("h2", { text: "Executed ComfyUI graph" }),
    el("h3", { text: "Exactly what was submitted to Comfy API v2, read from the saved *.graph.json" }))), tabs, view, bar, detail);
  let drag = null;
  view.onmousedown = e => { drag = { x: e.clientX, y: e.clientY, l: view.scrollLeft, t: view.scrollTop }; };
  window.addEventListener("mouseup", () => { drag = null; });
  window.addEventListener("mousemove", e => { if (drag) { view.scrollLeft = drag.l - (e.clientX - drag.x); view.scrollTop = drag.t - (e.clientY - drag.y); } });
  async function show(idx) {
    sel = idx; const g = graphs.find(x => x.idx === idx); if (!g) return;
    rec = await api.get(`/api/graph?run=${encodeURIComponent(run)}&file=${encodeURIComponent(g.file)}`);
    d = drawGraph(rec.graph, id => { d.select(id); nodeDetail(id); });
    const fit = Math.max(0.55, Math.min(1, (view.clientWidth - 8) / d.W));
    d.svg.setAttribute("width", d.W * fit); d.svg.setAttribute("height", d.H * fit);
    view.replaceChildren(d.svg); detail.hidden = true; drawTabs();
    bar.replaceChildren(...[el("div", {}, ...(g.models || []).map(m => el("span", { class: "chip", text: m })), el("span", { class: "chip node", text: `${g.n_nodes} nodes` }),
      g.job_id ? el("span", { class: "chip", text: "job " + g.job_id }) : null, g.worker ? el("span", { class: "chip", text: g.worker }) : null),
      el("div", { class: "row" }, el("button", { class: "btn sm ghost", type: "button", onclick: () => { d.svg.setAttribute("width", d.W); d.svg.setAttribute("height", d.H); } , text: "100%" }),
        el("button", { class: "btn sm ghost", type: "button", onclick: () => { const f = Math.max(0.55, Math.min(1, (view.clientWidth - 8) / d.W)); d.svg.setAttribute("width", d.W * f); d.svg.setAttribute("height", d.H * f); }, text: "Fit" }),
        el("a", { class: "btn sm", href: URL.createObjectURL(new Blob([JSON.stringify(rec.graph, null, 1)], { type: "application/json" })), download: g.file.split("/").pop().replace(".graph", "_api"), text: "Download API JSON" })),
      rec.note ? el("div", { class: "note", style: { width: "100%" }, text: rec.note }) : null].filter(Boolean));
  }
  function nodeDetail(id) {
    const n = rec.graph[id];
    detail.hidden = false;
    detail.replaceChildren(el("b", { text: `${n.class_type}  (node ${id})` }), el("dl", {}, ...Object.entries(n.inputs || {}).flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: Array.isArray(v) ? `-> node ${v[0]} output ${v[1]}` : typeof v === "string" && v.length > 400 ? v.slice(0, 400) + "..." : String(v) })])));
  }
  function drawTabs() {
    tabs.replaceChildren(...graphs.map(g => el("button", { class: "gtab", type: "button", "aria-pressed": String(g.idx === sel), onclick: () => show(g.idx) }, el("i", { class: g.status }), `${String(g.idx).padStart(2, "0")} ${g.label}`)));
  }
  return { root,
    async update(job) {
      const nl = job.graphs || [], changed = JSON.stringify(nl.map(g => [g.idx, g.status])) !== JSON.stringify(graphs.map(g => [g.idx, g.status]));
      graphs = nl;
      if (!nl.length) { tabs.replaceChildren(); view.replaceChildren(el("div", { class: "ph", style: { height: "100%" }, text: job.status === "running" || job.status === "queued" ? "Waiting for the first graph to be submitted..." : "No *.graph.json files in this run" })); return; }
      const lv = job.live, running = job.status === "running";
      const liveIdx = lv && lv.graph ? (graphs.find(g => lv.graph.endsWith(g.file)) || {}).idx : null;
      if (running && liveIdx && liveIdx !== sel && (liveFile !== lv.graph)) { liveFile = lv.graph; await show(liveIdx); }
      else if (sel == null) await show(nl[nl.length - 1].idx);
      else if (changed) drawTabs();
      if (d && running && lv && liveIdx === sel && lv.node) {
        const p = d.live(lv.node, lv.steps ? `${lv.step}/${lv.steps}` : "");
        if (p) { const sc = d.svg.getAttribute("width") / d.W; view.scrollTo({ left: Math.max(0, p.x * sc - 120), top: Math.max(0, p.y * sc - 120), behavior: "smooth" }); }
      } else if (d) d.live(null);
    } };
}

/* ---------- render page ---------- */
const optsOf = c => typeof c.options === "string" ? (LIB[c.options] || []) : (c.options || []).map(o => typeof o === "string" ? { v: o, l: o } : o);
const visible = c => !c.show || Object.entries(c.show).every(([k, vs]) => [].concat(vs).includes(FORM[k]));
function controlEl(c) {
  let inp;
  if (c.type === "select") inp = el("select", { id: "f_" + c.id, onchange: e => { FORM[c.id] = e.target.value; refresh(); } }, ...optsOf(c).map(o => el("option", { value: o.v, text: o.l || o.v })));
  else if (c.type === "textarea") inp = el("textarea", { id: "f_" + c.id, placeholder: c.placeholder || "", oninput: e => { FORM[c.id] = e.target.value; } });
  else if (c.type === "range") { const rv = el("span", { class: "rv", text: String(FORM[c.id]) }); inp = el("div", {}, el("input", { type: "range", id: "f_" + c.id, min: c.min, max: c.max, step: c.step, value: FORM[c.id], oninput: e => { FORM[c.id] = parseFloat(e.target.value); rv.textContent = e.target.value; } }), rv); }
  else if (c.type === "toggle") inp = el("label", { class: "tog" }, el("input", { type: "checkbox", id: "f_" + c.id, checked: !!FORM[c.id], onchange: e => { FORM[c.id] = e.target.checked; refresh(); } }), el("span", { text: c.help || "" }));
  else inp = el("input", { type: c.type === "number" ? "number" : "text", id: "f_" + c.id, min: c.min, max: c.max, step: c.step, placeholder: c.placeholder || "", value: FORM[c.id] ?? "", oninput: e => { FORM[c.id] = c.type === "number" ? parseFloat(e.target.value) : e.target.value; } });
  if (c.type === "select" || c.type === "textarea") inp.value = FORM[c.id] ?? "";
  return el("div", { class: "field" + (c.wide ? " wide" : ""), "data-c": c.id }, c.type === "toggle" ? null : el("label", { for: "f_" + c.id, text: c.label }), inp, c.help && c.type !== "toggle" ? el("div", { class: "help", text: c.help }) : null);
}
function uploadEl(u) {
  const slot = el("div", { class: "ph", style: { width: "56px", height: "56px", aspectRatio: "1", fontSize: "10px", flex: "none" }, text: "none" });
  const inp = el("input", { type: "file", accept: u.accept || "image/*", onchange: e => {
    const f = e.target.files[0]; FILES[u.id] = f || null; slot.replaceChildren();
    if (f) { slot.className = ""; slot.append(el(f.type.startsWith("video") ? "video" : "img", { src: URL.createObjectURL(f), muted: true })); } else { slot.className = "ph"; slot.textContent = "none"; }
  } });
  return el("div", { class: "field wide", "data-u": u.id }, el("label", { text: u.label }), el("label", { class: "drop" }, slot, el("div", {}, el("div", { text: "Choose file" }), el("div", { class: "help", text: u.help || "" })), inp));
}
function refresh() {
  document.querySelectorAll("[data-c]").forEach(n => { const c = CFG.controls.find(x => x.id === n.dataset.c); n.hidden = !visible(c); });
  document.querySelectorAll("[data-u]").forEach(n => { const u = CFG.uploads.find(x => x.id === n.dataset.u); n.hidden = u.show ? !Object.entries(u.show).every(([k, vs]) => [].concat(vs).includes(FORM[k])) : false; });
}
function formParams() { const p = {}; CFG.controls.forEach(c => { if (visible(c)) p[c.id] = FORM[c.id]; }); return p; }

function stackPanel() {
  return el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("div", {}, el("h2", { text: "The Comfy stack behind Render" }), el("h3", { text: "Open weights, run through Comfy API v2" }))),
    el("div", { class: "stack" }, ...CFG.stack.map(s => el("div", { class: "st" }, el("b", { text: s.stage }), el("div", { text: s.model }), s.nodes ? el("div", { class: "nd", text: s.nodes }) : null, s.why ? el("div", { class: "why", text: s.why }) : null))),
    el("div", { style: { marginTop: "12px" } }, ...CFG.nodes.map(n => el("span", { class: "chip node", title: n.what, text: n.name }))));
}

async function studio() {
  const main = $("#main");
  const plan = el("div", {}), recent = el("div", { class: "gal" });
  const go = el("button", { class: "btn primary", type: "button", text: CFG.render_label || "Render" });
  const groups = [...new Set(CFG.controls.map(c => c.group || "Settings"))];
  const form = el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", { text: CFG.form_title || "Settings" })),
    ...CFG.uploads.map(uploadEl),
    ...groups.map(g => el("div", { class: "grid2" }, el("div", { class: "fset", text: g }), ...CFG.controls.filter(c => (c.group || "Settings") === g).map(controlEl))),
    el("div", { class: "row" }, go, CFG.plan ? el("button", { class: "btn", type: "button", text: "Preview shot list", onclick: async () => {
      try { const r = await api.post0("/api/plan", formParams()); plan.replaceChildren(el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", { text: "Shot list (from the pack's rig library, no GPU)" })), ...r.map(s => el("div", { class: "shot" }, el("b", { text: s.label }), el("div", { class: "p", text: s.motion_prompt })))));
      } catch (e) { toast(e.message, true); } } }) : null,
      CFG.dry ? el("span", { class: "chip", text: "dry-run mode: placeholder outputs, no ComfyUI" }) : null,
      CFG.replay ? el("span", { class: "chip", text: "replaying a finished render (" + CFG.replay + "): cached, no GPU" }) : null));
  main.replaceChildren(el("h1", { text: CFG.name }), el("p", { class: "lead", text: CFG.tagline }), el("div", { class: "cols" }, el("div", {}, form, plan), el("div", {}, stackPanel(), el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", { text: "Recent runs" }), el("a", { class: "btn sm ghost", href: "#/gallery", text: "All" })), recent))));
  refresh();
  go.onclick = async () => {
    const fd = new FormData(), names = [];
    CFG.uploads.forEach(u => { if (FILES[u.id]) { fd.append("files", FILES[u.id]); names.push(u.id); } });
    fd.append("params", JSON.stringify(formParams())); fd.append("fields", names.join(","));
    go.disabled = true;
    try { const r = await api.post("/api/jobs", fd); location.hash = "#/run/" + r.run; } catch (e) { toast(e.message, true); } finally { go.disabled = false; }
  };
  api.get("/api/runs").then(rs => recent.replaceChildren(...rs.slice(0, 6).map(runTile))).catch(() => {});
}
api.post0 = async (u, obj) => { const r = await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) }); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); };

function runTile(r) {
  const media = r.reel ? el("video", { src: vurl(r.reel), muted: true, loop: true, preload: "metadata", onmouseenter: e => e.target.play(), onmouseleave: e => e.target.pause() })
    : r.poster ? (isVid(r.poster) ? el("video", { src: vurl(r.poster), muted: true, preload: "metadata" }) : el("img", { src: runUrl(r.poster), loading: "lazy" })) : el("div", { class: "ph", text: r.status });
  return el("a", { class: "tile", href: "#/run/" + r.run }, media, el("div", { class: "cap" }, el("span", { text: r.title }), el("span", { text: `${r.videos} clip${r.videos === 1 ? "" : "s"}` })));
}
async function gallery() {
  const main = $("#main"), box = el("div", { class: "gal big" });
  main.replaceChildren(el("h1", { text: "Gallery" }), el("p", { class: "lead", text: "Every run under runs/, including ones made with the command line. Open one to see its plates, clips and the executed ComfyUI graphs." }), box);
  const rs = await api.get("/api/runs");
  box.replaceChildren(...(rs.length ? rs.map(runTile) : [el("div", { class: "ph", text: "No runs yet. Render something first." })]));
}

/* ---------- run page ---------- */
function mediaTile(url, cap) {
  const m = isVid(url) ? el("video", { src: vurl(url), muted: true, loop: true, preload: "metadata", controls: false, onmouseenter: e => e.target.play(), onmouseleave: e => e.target.pause() }) : el("img", { src: runUrl(url), loading: "lazy" });
  return el("div", { class: "tile", onclick: () => lightbox(runUrl(url)) }, m, el("div", { class: "cap" }, el("span", { text: cap || url.split("/").pop() })));
}
function resultsPanel(f) {
  const parts = [], root = f.root || [];
  const reel = root.find(x => x.endsWith("reel.mp4")), sheet = (f.plates || []).find(x => x.endsWith("contact_sheet.jpg"));
  if (reel) parts.push(el("h3", { text: "Reel" }), el("div", { class: "gal big" }, mediaTile(reel, "reel.mp4 (hard-cut, LTX-2.5 audio)")));
  const other = root.filter(x => x !== reel);
  if (other.length) parts.push(el("h3", { text: "Outputs", style: { marginTop: "16px" } }), el("div", { class: "gal big" }, ...other.map(x => mediaTile(x))));
  if (sheet) parts.push(el("h3", { text: "Plates contact sheet", style: { marginTop: "16px" } }), el("div", { class: "gal big" }, mediaTile(sheet, "contact_sheet.jpg")));
  const clips = (f.clips || []).filter(x => /_shot\.mp4$/.test(x)); if (clips.length) parts.push(el("h3", { text: `Clips (${clips.length}, speed/impact FX applied, audio kept)`, style: { marginTop: "16px" } }), el("div", { class: "gal" }, ...clips.map(x => mediaTile(x))));
  const plates = (f.plates || []).filter(x => !x.endsWith("contact_sheet.jpg")); if (plates.length) parts.push(el("h3", { text: `Plates (${plates.length})`, style: { marginTop: "16px" } }), el("div", { class: "gal" }, ...plates.map(x => mediaTile(x))));
  const refs = f.refs || []; if (refs.length) parts.push(el("h3", { text: "Identity references", style: { marginTop: "16px" } }), el("div", { class: "gal" }, ...refs.map(x => mediaTile(x))));
  return el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", { text: "Results" })), ...(parts.length ? parts : [el("div", { class: "ph", text: "Nothing rendered yet" })]));
}
async function runPage(run) {
  const main = $("#main"), gp = graphPanel(run);
  const head = el("div", { class: "row", style: { justifyContent: "space-between", marginBottom: "14px" } }), live = el("div", {}), results = el("div", {}), logBox = el("pre", { class: "log" });
  main.replaceChildren(head, live, results, gp.root, el("div", { class: "panel" }, el("div", { class: "panel-head" }, el("h2", { text: "Runner log" })), logBox));
  function paint(j) {
    const act = j.status === "running" || j.status === "queued";
    head.replaceChildren(...[el("div", {}, el("h1", { text: j.params && j.params.title || run.split("/").pop() }), el("span", { class: "chip " + (j.status === "done" ? "ok" : j.status === "failed" ? "bad" : "run"), text: j.status + (j.queue_position ? ` (queue ${j.queue_position})` : "") }), CFG.replay ? el("span", { class: "chip", style: { marginLeft: "8px" }, text: "replaying a finished render, nothing submitted to ComfyUI" }) : null),
      act && j.status !== "done" ? el("button", { class: "btn", type: "button", text: "Cancel", onclick: () => api.post("/api/cancel?run=" + encodeURIComponent(run)) }) : null].filter(Boolean));
    const lv = j.live;
    live.replaceChildren(...(act || j.error ? [el("div", { class: "live" }, el("div", { class: "t", text: j.stage || "starting" }),
      el("div", { class: "bar" }, el("i", { style: { width: Math.round((j.progress || 0) * 100) + "%" } })),
      lv && lv.node ? el("div", { class: "s", text: `node ${lv.node} ${lv.cls || ""}${lv.steps ? `  step ${lv.step}/${lv.steps}` : ""}  nodes ${lv.nodes_done ?? "?"}/${lv.nodes_total ?? "?"}` }) : el("div", { class: "s", text: lv && lv.status ? `ComfyUI job: ${lv.status}` : `${Math.round((j.progress || 0) * 100)}%` }),
      j.error ? el("pre", { class: "log", text: j.error }) : null)] : []));
    results.replaceChildren(resultsPanel(j.files || {}));
    logBox.textContent = (j.log || []).join("\n"); logBox.scrollTop = logBox.scrollHeight;
    gp.update(j);
  }
  paint(await api.get("/api/run?run=" + encodeURIComponent(run)));
  const es = new EventSource("/api/events?run=" + encodeURIComponent(run));
  es.addEventListener("job", e => paint(JSON.parse(e.data)));
  es.onerror = () => es.close();
  cleanup = () => es.close();
}

/* ---------- shell ---------- */
async function route() {
  cleanup(); cleanup = () => {};
  const h = location.hash.slice(2) || "studio", [page, ...rest] = h.split("/");
  document.querySelectorAll(".nav a").forEach(a => a.classList.toggle("active", a.dataset.r === (page === "run" ? "gallery" : page)));
  $("#topbar").replaceChildren(el("div", { text: CFG.topline }), el("div", { id: "cstat" }));
  comfyStat();
  try { if (page === "gallery") await gallery(); else if (page === "run") await runPage(decodeURIComponent(rest.join("/"))); else await studio(); } catch (e) { toast(e.message, true); }
  window.scrollTo(0, 0);
}
async function comfyStat() {
  try { const c = await api.get("/api/comfy"); const t = el("span", {}, el("i", { class: "dot " + (c.up ? "on" : "off") }), c.up ? `Comfy API v2 proxy ${c.url}` : `Comfy API v2 proxy not reachable (${c.url})`);
    $("#cstat") && $("#cstat").replaceChildren(t); $("#foot").replaceChildren(t.cloneNode(true), el("div", { text: CFG.replay ? "replaying a finished render" : CFG.dry ? "dry-run mode" : "live mode" })); } catch (e) { /* ignore */ }
}
(async () => {
  CFG = await api.get("/api/config"); LIB = await api.get("/api/presets");
  document.documentElement.style.setProperty("--accent", CFG.accent); document.documentElement.style.setProperty("--accent-ink", CFG.accent_ink || "#111");
  const rgb = CFG.accent.match(/\w\w/g).map(h => parseInt(h, 16)); document.documentElement.style.setProperty("--accent-soft", `rgba(${rgb},.12)`); document.documentElement.style.setProperty("--accent-line", `rgba(${rgb},.45)`);
  document.title = CFG.name; $("#brand").textContent = CFG.name;
  CFG.controls.forEach(c => { FORM[c.id] = c.default ?? (c.type === "toggle" ? false : ""); });
  window.addEventListener("hashchange", route); route(); setInterval(comfyStat, 8000);
})();
