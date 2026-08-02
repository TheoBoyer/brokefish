// Replay and derivation, `docs/debugger.md` §5 and §3.
//
// The trace stores per-simulation deltas and nothing that can be computed from
// them, so this module is where the tree comes back: `replay` rebuilds the state
// after any prefix, and `puct` derives §6.6's score from that state.
//
// Everything here is float64 against a search that ran in float32, which is §3's
// caveat: visit counts agree exactly, `Q` to about 1e-7, and a displayed ordering
// between two nearly equal scores is not the ordering the search guaranteed.

export const FORMAT = 1;

export function width(node) {
  return node.edges ? node.edges.move.length : 0;
}

// The tree after the first `k` simulations (all of them when `k` is undefined).
// Backwards is a replay from zero rather than an undo, because a running mean
// does not invert stably.
export function replay(trace, k) {
  const root = trace.root;
  const nodes = new Map([[0, root]]);
  const N = new Map([[0, new Int32Array(width(root))]]);
  const Q = new Map([[0, new Float64Array(width(root))]]);
  const child = new Map([[0, new Int32Array(width(root)).fill(-1)]]);

  const sims = trace.sims.slice(0, k === undefined ? trace.sims.length : k);
  for (const sim of sims) {
    const rec = sim.created;
    if (rec) {
      nodes.set(rec.id, rec);
      N.set(rec.id, new Int32Array(width(rec)));
      Q.set(rec.id, new Float64Array(width(rec)));
      child.set(rec.id, new Int32Array(width(rec)).fill(-1));
      child.get(rec.parent)[rec.pedge] = rec.id;
    }
    const L = sim.path.length;
    for (let d = 0; d < L; d++) {
      const [v, e] = sim.path[d];
      // `mcts.md` §6.5. Inverting this parity shows the search preferring its
      // worst moves, and at L = 2 the wrong parity agrees with the right one.
      const q = (L - d) % 2 === 0 ? sim.value : 1 - sim.value;
      const n = ++N.get(v)[e];
      const qq = Q.get(v);
      qq[e] += (q - qq[e]) / n;
    }
  }
  return { nodes, N, Q, child };
}

// §6.6, split into its two terms because almost every "why that move" resolves
// into one of them dominating and a combined score hides which.
export function scores(trace, state, node) {
  const c = trace.config;
  const n = state.N.get(node), q = state.Q.get(node);
  const prior = state.nodes.get(node).edges.prior;
  let nv = 0;
  for (const x of n) nv += x;
  let pbc = Math.log((nv + c.pb_c_base + 1) / c.pb_c_base) + c.pb_c_init;
  pbc *= Math.sqrt(nv);
  return Array.from(n, (ni, i) => {
    const u = (pbc * prior[i]) / (1 + ni);
    // First-play urgency: an unvisited edge takes 0.5, the draw. In the [0,1]
    // convention 0 is a certain loss, so scoring it 0 would make every
    // unexplored move look lost -- `torch_impl.FPU_DRAW`, spec §6.6. The table
    // still shows it blank, since it is an assumption and not a measured Q.
    const qi = ni > 0 ? q[i] : 0.5;
    return { u, q: ni > 0 ? q[i] : null, score: u + qi, N: ni, P: prior[i] };
  });
}

// Which simulations passed through each node, built once at load. This is what
// makes "abandoned after simulation 200" a visible fact.
export function visitIndex(trace) {
  const byNode = new Map();
  trace.sims.forEach((sim, i) => {
    for (const [v] of sim.path) {
      if (!byNode.has(v)) byNode.set(v, []);
      const list = byNode.get(v);
      if (list[list.length - 1] !== i) list.push(i);
    }
    if (!byNode.has(sim.leaf)) byNode.set(sim.leaf, []);
  });
  return byNode;
}

// §5's own check: the replayed final state against the recorded tail. A viewer
// that silently disagrees with the search it displays is worse than no viewer.
export function checkFinal(trace, state) {
  const n = state.N.get(0), q = state.Q.get(0);
  const want = trace.final_root;
  for (let i = 0; i < want.N.length; i++) {
    if (n[i] !== want.N[i]) return `root edge ${i}: replayed N ${n[i]}, recorded ${want.N[i]}`;
    if (Math.abs(q[i] - want.Q[i]) > 1e-5) {
      return `root edge ${i}: replayed Q ${q[i]}, recorded ${want.Q[i]}`;
    }
  }
  return null;
}

// §7.4. Not per-simulation: these are properties of the whole run.
export function summary(trace) {
  const depths = trace.sims.map((s) => s.path.length);
  const hist = new Map();
  for (const d of depths) hist.set(d, (hist.get(d) || 0) + 1);
  const codes = new Map();
  let created = 0, maxLegal = 0, dropped = 0, truncated = 0;
  for (const sim of trace.sims) {
    if (!sim.created) continue;
    created++;
    const c = sim.created;
    codes.set(c.code, (codes.get(c.code) || 0) + 1);
    if (c.edges) {
      maxLegal = Math.max(maxLegal, c.n_legal);
      dropped += c.truncated_mass;
      if (c.n_legal > c.edges.move.length) truncated++;
    }
  }
  const n = trace.final_root.N, q = trace.final_root.Q;
  const argmax = (a) => a.reduce((b, x, i) => (x > a[b] ? i : b), 0);
  return {
    nodes: created + 1,
    pool: trace.config.n + 1,
    depthHist: [...hist.entries()].sort((a, b) => a[0] - b[0]),
    maxDepth: Math.max(...depths),
    codes: [...codes.entries()].sort((a, b) => a[0] - b[0]),
    maxLegal: Math.max(maxLegal, trace.root.n_legal || 0),
    truncated,
    truncatedMass: dropped + (trace.root.truncated_mass || 0),
    E: trace.config.E,
    // §7.4: visits and Q disagreeing at the root is the cheapest visible symptom
    // of a `pb_c_init` tuned for the wrong value convention.
    agrees: argmax(n) === argmax(q),
    bestVisits: argmax(n),
    bestQ: argmax(q),
  };
}
