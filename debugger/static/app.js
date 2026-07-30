// The viewer. `docs/debugger.md` §7.
//
// Every view reads the replayed state at the current simulation index `k`, which
// is global to the page: changing `k` changes everything at once. No chess logic
// runs here. Boards are rendered by the server through python-chess and moves
// arrive from it already named, so the browser never learns the rules.

import { replay, scores, visitIndex, checkFinal, summary, width, FORMAT } from "/static/trace.js";

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid !== null && kid !== undefined) node.append(kid.nodeType ? kid : String(kid));
  }
  return node;
};
const f = (x, d = 4) => (x === null || x === undefined ? "" : x.toFixed(d));
const status = (msg) => { $("#status").textContent = msg || ""; };

async function api(path, body, method) {
  const opts = body
    ? { method: method || "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify(body) }
    : { method: method || "GET" };
  const res = await fetch(path, opts);
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data && data.error ? data.error : res.statusText);
  return data;
}

function boardUrl(fen, lastmove, flipped, coordinates = true) {
  const q = new URLSearchParams({ fen });
  if (lastmove) q.set("lastmove", lastmove);
  if (flipped) q.set("flipped", "true");
  if (!coordinates) q.set("coordinates", "false");
  return "/api/board.svg?" + q;
}

// -- tabs ------------------------------------------------------------------ //

for (const b of document.querySelectorAll("nav button")) {
  b.onclick = () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("on", x === b));
    for (const name of ["traces", "viewer", "play"]) {
      $("#" + name).hidden = name !== b.dataset.tab;
    }
  };
}
const showTab = (name) => document.querySelector(`nav button[data-tab=${name}]`).click();

function banner(msg) {
  $("#banner").hidden = !msg;
  $("#banner").textContent = msg || "";
}

// Every handler that calls the API goes through this. A rejected fetch with no
// catch is invisible, and the failure that actually happens is a 404 on a game
// the server no longer has: sessions live in memory, so a restart loses them.
const guard = (fn) => async (...args) => {
  try { await fn(...args); } catch (err) { status(""); banner(String(err)); }
};

// -- the trace list -------------------------------------------------------- //

async function refreshTraces() {
  const rows = await api("/api/traces");
  const body = $("#trace-list tbody");
  body.replaceChildren(...rows.map((r) =>
    el("tr", {},
      el("td", { class: "l" }, el("a", { onclick: guard(() => openTrace(r.id)) }, r.id)),
      el("td", {}, r.kind),
      el("td", {}, r.n),
      el("td", {}, r.played),
      el("td", {}, r.checkpoint || "untrained"),
      el("td", { class: "l dim" }, r.fen),
      el("td", {}, el("a", {
        onclick: guard(async () => {
          await api(`/api/trace/${r.id}`, null, "DELETE");
          await refreshTraces();
        }),
      }, "delete")))));
}

$("#new-search").onsubmit = async (e) => {
  e.preventDefault();
  const d = Object.fromEntries(new FormData(e.target));
  status("searching…");
  try {
    const out = await api("/api/search", {
      fen: d.fen || null, n: +d.n, eps: +d.eps, seed: +d.seed,
      checkpoint: d.checkpoint || null,
    });
    status(`${out.seconds.toFixed(1)} s`);
    await refreshTraces();
    openTrace(out.trace_id);
  } catch (err) { status(""); banner(String(err)); }
};

$("#do-eval").onclick = async () => {
  const d = Object.fromEntries(new FormData($("#new-search")));
  status("evaluating…");
  try {
    const out = await api("/api/eval", { fen: d.fen || null,
                                         checkpoint: d.checkpoint || null });
    status("");
    $("#eval-out").replaceChildren(
      el("div", { class: "kv" },
        el("div", {}, "value [0,1]"), el("div", {}, f(out.value, 5)),
        el("div", {}, "value tanh"), el("div", {}, f(out.value_tanh, 5)),
        el("div", {}, "edges"), el("div", {}, out.n_edges)),
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", { class: "l" }, "move"), el("th", {}, "P"))),
        el("tbody", {}, out.top.map((t) =>
          el("tr", {}, el("td", { class: "l" }, t.san), el("td", {}, f(t.p, 5)))))));
  } catch (err) { status(""); banner(String(err)); }
};

// -- the viewer ------------------------------------------------------------ //

const V = { trace: null, k: 0, state: null, node: 0, visits: null, timer: null };

async function openTrace(id) {
  status("loading…");
  const trace = await api(`/api/trace/${id}`);
  if (trace.format !== FORMAT) {
    banner(`trace format ${trace.format}, this viewer reads ${FORMAT}`);
    return;
  }
  V.trace = trace;
  V.id = id;
  V.visits = visitIndex(trace);
  V.node = 0;
  $("#k").max = trace.sims.length;
  $("#k").value = trace.sims.length;
  status("");
  banner(null);
  showTab("viewer");
  setK(trace.sims.length);

  const c = trace.config;
  $("#trace-header").replaceChildren(el("div", { class: "kv" },
    el("div", {}, "trace"), el("div", {}, id),
    el("div", {}, "kind"), el("div", {}, `${trace.kind}, ${trace.created}`),
    el("div", {}, "config"),
    el("div", {}, `n=${c.n} E=${c.E} eps=${c.eps} alpha=${c.alpha} ` +
                  `pb_c=(${c.pb_c_base}, ${c.pb_c_init}) seed=${c.seed}`),
    el("div", {}, "impl"),
    el("div", {}, Object.entries(trace.impl).map(([k, v]) => `${k}=${v}`).join(" ")),
    el("div", {}, "checkpoint"),
    el("div", {}, trace.checkpoint && trace.checkpoint.path
        ? `${trace.checkpoint.path} gen ${trace.checkpoint.weight_gen}` : "untrained"),
    el("div", {}, "played"),
    el("div", {}, `${trace.root.edges.san[trace.played]} (${trace.temperature})`)));
  $("#run-summary").replaceChildren(renderSummary(trace));
}

function setK(k) {
  const trace = V.trace;
  V.k = Math.max(0, Math.min(k, trace.sims.length));
  V.state = replay(trace, V.k);
  $("#k").value = V.k;
  $("#k-label").textContent = `${V.k} / ${trace.sims.length}`;
  if (V.k === trace.sims.length) {
    const bad = checkFinal(trace, V.state);
    banner(bad ? `replay disagrees with the recorded final root: ${bad}` : null);
  }
  if (!V.state.nodes.has(V.node)) V.node = 0;
  renderRoot();
  renderSim();
  renderNode(V.node);
}

$("#k").oninput = (e) => setK(+e.target.value);
for (const b of document.querySelectorAll("#scrubber [data-step]")) {
  b.onclick = () => setK(V.k + +b.dataset.step);
}
$("#play-sims").onclick = (e) => {
  if (V.timer) { clearInterval(V.timer); V.timer = null; e.target.textContent = "play"; return; }
  e.target.textContent = "stop";
  V.timer = setInterval(() => {
    if (V.k >= V.trace.sims.length) { $("#play-sims").click(); return; }
    setK(V.k + 1);
  }, 120);
};

// §7.1. One row per edge, sorted by visits, ties by edge index.
function edgeTable(node, onPath) {
  const trace = V.trace, state = V.state;
  const rec = state.nodes.get(node);
  if (!rec.edges) return el("div", { class: "dim" }, "terminal node, no edges");
  const rows = scores(trace, state, node);
  const total = rows.reduce((a, r) => a + r.N, 0) || 1;
  const child = state.child.get(node);
  const order = rows.map((_, i) => i)
    .sort((a, b) => rows[b].N - rows[a].N || a - b);
  return el("table", {},
    el("thead", {}, el("tr", {},
      el("th", { class: "l" }, "move"), el("th", {}, "N"), el("th", {}, "pi"),
      el("th", {}, "Q"), el("th", {}, "Q [-1,1]"), el("th", {}, "P"),
      el("th", {}, "U"), el("th", {}, "Q+U"), el("th", {}, "child"))),
    el("tbody", {}, order.map((i) => {
      const r = rows[i];
      const kid = child[i];
      return el("tr", { class: onPath === i ? "onpath sel" : "" },
        el("td", { class: "l" }, rec.edges.san[i]),
        el("td", {}, r.N),
        el("td", {}, f(r.N / total, 3)),
        // §7.1: an unvisited edge shows a blank, since a 0 would read as "this
        // move was estimated as a loss" where the truth is "no estimate".
        r.q === null ? el("td", { class: "blank" }, "-") : el("td", {}, f(r.q)),
        r.q === null ? el("td", { class: "blank" }, "-") : el("td", {}, f(2 * r.q - 1, 3)),
        el("td", {}, f(r.P, 4)),
        el("td", {}, f(r.u, 4)),
        el("td", {}, f(r.score, 4)),
        el("td", {}, kid >= 0
          ? el("a", { onclick: () => renderNode(kid) }, kid)
          : ""));
    })));
}

function renderRoot() {
  const rec = V.state.nodes.get(0);
  const sim = V.trace.sims[V.k - 1];
  const onPath = V.k > 0 ? sim.path[0][1] : null;
  $("#root-sub").textContent = `${width(rec)} edges, ${rec.n_legal} legal`;
  $("#root-table").replaceChildren(edgeTable(0, onPath));
}

// §7.2. The path with the score that decided each step, recomputed from the
// state *before* the simulation was applied.
function renderSim() {
  const trace = V.trace;
  if (V.k === 0) {
    $("#sim-detail").replaceChildren(el("div", { class: "dim" }, "before simulation 1"));
    showBoard(trace.root.fen, null, `root: ${trace.root.fen}`);
    return;
  }
  const sim = trace.sims[V.k - 1];
  const before = replay(trace, V.k - 1);
  const rows = [];
  let last = null;
  for (const [v, e] of sim.path) {
    const sc = scores(trace, before, v);
    const order = sc.map((_, i) => i).sort((a, b) => sc[b].score - sc[a].score || a - b);
    const margin = sc.length > 1 ? sc[order[0]].score - sc[order[1]].score : Infinity;
    const rec = before.nodes.get(v);
    rows.push({
      node: v, edge: e, san: rec.edges.san[e],
      score: sc[e].score, q: sc[e].q, u: sc[e].u,
      runnerUp: order[1] === undefined ? null : rec.edges.san[order[1]],
      margin, agrees: order[0] === e,
    });
    last = e;
  }
  const leaf = sim.created || V.state.nodes.get(sim.leaf);
  showBoard(leaf.fen, leaf.parent >= 0 ? uciOf(leaf) : null,
            `leaf ${sim.leaf}, depth ${leaf.depth}: ${leaf.fen}`);
  $("#sim-detail").replaceChildren(
    el("div", { class: "kv" },
      el("div", {}, "simulation"), el("div", {}, `${sim.s} of ${trace.sims.length - 1}`),
      el("div", {}, "ended"), el("div", {}, sim.ended),
      el("div", {}, "leaf value"),
      el("div", {}, `${f(sim.value, 5)}  (tanh ${f(2 * sim.value - 1, 4)}, ` +
                    `${leaf.value_source})`),
      el("div", {}, "line"),
      el("div", {}, rows.map((r) => r.san).join(" "))),
    el("table", {},
      el("thead", {}, el("tr", {},
        el("th", { class: "l" }, "d"), el("th", { class: "l" }, "node"),
        el("th", { class: "l" }, "move"), el("th", {}, "Q"), el("th", {}, "U"),
        el("th", {}, "Q+U"), el("th", { class: "l" }, "runner-up"),
        el("th", {}, "margin"))),
      el("tbody", {}, rows.map((r, d) =>
        el("tr", {},
          el("td", { class: "l" }, d),
          el("td", { class: "l" }, el("a", { onclick: () => renderNode(r.node) }, r.node)),
          el("td", { class: "l" }, r.san),
          r.q === null ? el("td", { class: "blank" }, "-") : el("td", {}, f(r.q)),
          el("td", {}, f(r.u)),
          el("td", {}, f(r.score)),
          el("td", { class: "l dim" }, r.runnerUp || ""),
          // §3: the viewer's float64 ordering is not the search's float32
          // guarantee, so a margin this small is marked rather than trusted.
          el("td", { class: r.margin < 1e-5 ? "warn" : "" },
             r.margin === Infinity ? "" : f(r.margin, 6)))))));
  void last;
}

function uciOf(rec) {
  const parent = V.state.nodes.get(rec.parent);
  return parent && parent.edges ? parent.edges.uci[rec.pedge] : null;
}

// §7.3.
function renderNode(id) {
  V.node = id;
  const state = V.state;
  const rec = state.nodes.get(id);
  if (!rec) {
    $("#node-detail").replaceChildren(
      el("div", { class: "dim" }, `node ${id} does not exist yet at k = ${V.k}`));
    return;
  }
  const seen = (V.visits.get(id) || []).filter((i) => i < V.k);
  const CODES = ["live", "checkmate", "stalemate", "fifty-move", "repetition",
                 "insufficient"];
  $("#node-sub").textContent = `${id}`;
  $("#node-detail").replaceChildren(
    el("div", { class: "kv" },
      el("div", {}, "fen"), el("div", {}, rec.fen),
      el("div", {}, "depth / parent"),
      el("div", {}, `${rec.depth} / ${rec.parent < 0 ? "root" : rec.parent}`),
      el("div", {}, "terminal"),
      el("div", { class: rec.code ? "warn" : "" }, `${CODES[rec.code] || rec.code}`),
      el("div", {}, "value"),
      el("div", {}, `${f(rec.value, 5)} (tanh ${f(2 * rec.value - 1, 4)}, ${rec.value_source})`),
      el("div", {}, "repetition"), el("div", {}, `${rec.rep_raw} raw, feature ${rec.rep}`),
      el("div", {}, "hash"), el("div", {}, rec.hash),
      el("div", {}, "legal / edges"),
      el("div", {}, rec.edges ? `${rec.n_legal} / ${width(rec)}` +
        (rec.n_legal > width(rec) ? `  truncated, ${f(rec.truncated_mass, 4)} mass` : "")
        : "-"),
      el("div", {}, "visited by"),
      el("div", {}, seen.length
        ? `${seen.length} simulations, first ${seen[0]}, last ${seen[seen.length - 1]}`
        : "not yet")),
    el("div", { class: "row" },
      el("a", { onclick: () => showBoard(rec.fen, uciOf(rec), `node ${id}: ${rec.fen}`) },
         "show board"),
      " ",
      seen.length ? el("a", { onclick: () => setK(seen[0] + 1) }, "  jump to first visit") : null),
    edgeTable(id, null));
}

function showBoard(fen, lastmove, caption) {
  $("#board").replaceChildren(el("img", { src: boardUrl(fen, lastmove) }));
  $("#board-caption").textContent = caption;
}

// §7.4.
function renderSummary(trace) {
  const s = summary(trace);
  const max = Math.max(...s.depthHist.map(([, c]) => c));
  const CODES = ["live", "checkmate", "stalemate", "fifty-move", "repetition",
                 "insufficient"];
  return el("div", {},
    el("div", { class: "kv" },
      el("div", {}, "nodes / pool"),
      el("div", {}, `${s.nodes} / ${s.pool}`),
      el("div", {}, "max depth"), el("div", {}, s.maxDepth),
      el("div", {}, "max legal / E"),
      el("div", { class: s.maxLegal > s.E ? "warn" : "" }, `${s.maxLegal} / ${s.E}`),
      el("div", {}, "truncated nodes"),
      el("div", {}, `${s.truncated}, mass ${f(s.truncatedMass, 4)}`),
      el("div", {}, "terminal children"),
      el("div", {}, s.codes.filter(([c]) => c !== 0)
        .map(([c, n]) => `${CODES[c] || c}: ${n}`).join("  ") || "none"),
      // §7.4: visits and Q disagreeing at the root is the cheapest visible
      // symptom of an exploration constant tuned for the wrong Q range.
      el("div", {}, "argmax N = argmax Q"),
      el("div", { class: s.agrees ? "good" : "warn" },
         s.agrees ? "yes" : `no: visits ${trace.root.edges.san[s.bestVisits]}, ` +
                            `Q ${trace.root.edges.san[s.bestQ]}`)),
    el("table", {},
      el("thead", {}, el("tr", {}, el("th", { class: "l" }, "depth"),
                          el("th", {}, "sims"), el("th", { class: "l" }, ""))),
      el("tbody", {}, s.depthHist.map(([d, c]) =>
        el("tr", {}, el("td", { class: "l" }, d), el("td", {}, c),
           el("td", { class: "l" },
              el("span", { class: "bar", style: `width:${(120 * c) / max}px` }, "")))))));
}

// -- play ------------------------------------------------------------------ //

const G = { id: null, state: null, from: null };

$("#new-game").onsubmit = async (e) => {
  e.preventDefault();
  const d = Object.fromEntries(new FormData(e.target));
  status("starting…");
  try {
    const out = await api("/api/game", {
      checkpoint: d.checkpoint || null, n: +d.n, human_color: d.human_color,
      fen: d.fen || null,
    });
    G.id = out.game_id;
    setGame(out);
    status("");
  } catch (err) { status(""); banner(String(err)); }
};

$("#takeback").onclick = guard(async () => {
  if (!G.id) return;
  setGame(await api(`/api/game/${G.id}/takeback`, {}));
  banner(null);
});

$("#search-here").onclick = guard(async () => {
  if (!G.state) return;
  const n = +new FormData($("#new-game")).get("n");
  status("searching…");
  const out = await api("/api/search", { fen: G.state.fen, n, eps: 0.0 });
  status(`${out.seconds.toFixed(1)} s`);
  await refreshTraces();
  openTrace(out.trace_id);
});

function setGame(state) {
  G.state = state;
  G.from = null;
  const flipped = !state.human_white;
  const CODES = ["", "checkmate", "stalemate", "fifty-move draw", "repetition draw",
                 "insufficient material"];
  $("#game-board").replaceChildren(
    // No coordinates: the overlay below covers the image's own box, and a
    // coordinate margin would shift every square by half a file.
    el("img", { src: boardUrl(state.fen, state.last, flipped, false) }),
    el("div", { id: "squares" }, Array.from({ length: 64 }, (_, i) => {
      const row = Math.floor(i / 8), col = i % 8;
      const square = flipped ? row * 8 + (7 - col) : (7 - row) * 8 + col;
      return el("div", { "data-sq": square, onclick: () => clickSquare(square) });
    })));
  $("#game-caption").textContent =
    `${state.fen}${state.code ? "   " + CODES[state.code] : ""}`;
  const links = state.traces.map((t, i) =>
    t ? el("div", {}, `ply ${i + 1} `,
           el("a", { onclick: guard(() => openTrace(t)) }, t)) : null);
  $("#game-moves").replaceChildren(
    el("div", { class: "dim" }, `ply ${state.ply}, ${state.legal.length} legal moves`),
    ...links.filter(Boolean).reverse());
}

const FILES = "abcdefgh";
const nameOf = (sq) => FILES[sq % 8] + (Math.floor(sq / 8) + 1);

async function clickSquare(square) {
  if (!G.state || G.state.code) return;
  const legal = G.state.legal;
  if (G.from === null) {
    if (!legal.some((u) => u.startsWith(nameOf(square)))) return;
    G.from = square;
    highlight();
    return;
  }
  const base = nameOf(G.from) + nameOf(square);
  const options = legal.filter((u) => u.startsWith(base));
  G.from = null;
  highlight();
  if (!options.length) return;
  let uci = options[0];
  if (options.length > 1) {
    // The engine's action space carries the promotion choice on the edge, so a
    // promotion is four moves here and the human picks which.
    const pick = prompt(`promote to? ${options.map((u) => u.slice(4)).join(" ")}`, "q");
    uci = options.find((u) => u.endsWith(pick)) || options[0];
  }
  status("thinking…");
  try {
    setGame(await api(`/api/game/${G.id}/move`, { uci }));
    status("");
    refreshTraces();
  } catch (err) { status(""); banner(String(err)); }
}

function highlight() {
  for (const d of document.querySelectorAll("#squares div")) {
    const sq = +d.dataset.sq;
    d.classList.toggle("from", G.from === sq);
    d.classList.toggle("to", G.from !== null &&
      G.state.legal.some((u) => u === nameOf(G.from) + nameOf(sq) ||
                                u.startsWith(nameOf(G.from) + nameOf(sq))));
  }
}

refreshTraces().catch((err) => banner(String(err)));
