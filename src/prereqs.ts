// Prerequisite edges between posts: `[a, b]` means "read a before b". Post
// pages render "Before reading this" / "Read next" lines from these, and the
// concept map (/map.html) draws them as arrows.
//
// Refs follow the same rules as src/series.ts: a bare slug (`vaes`), or the
// full filename stem (`2024-08-18-ssl`) when two posts share a slug. Note that
// `hopfield_from_scratch` is the Cover's universal portfolio post and `up` is
// the from-scratch Hopfield one (the filenames are swapped).
//
// Keep this a meaningful DAG, not an exhaustive one: an edge should be a post
// whose ideas the later post actually builds on.

export type Prereq = [before: string, after: string];

export const prereqs: Prereq[] = [
  // Probability and information theory underneath the generative models.
  ["prob", "divergences"],
  ["prob", "kalman"],
  ["prob", "hopfield_from_scratch"],
  ["divergences", "vaes"],
  ["kolmogorov", "epiplexity"],
  ["divergences", "epiplexity"],

  // Generative modelling: VAEs to diffusion, samplers, guidance and flows.
  ["vaes", "diffusion"],
  ["diffusion", "ddim_ddpm"],
  ["ddim_ddpm", "cfg"],
  ["diffusion", "flow"],
  ["diffusion", "timeseries-diffusion"],

  // Transformers and what is built on them.
  ["transformer", "bertimbau"],
  ["transformer", "llm-archs"],
  ["transformer", "decoding"],
  ["llm-archs", "moe"],
  ["llm-archs", "rag"],
  ["llm-archs", "llm-quant"],
  ["moe", "deepseek"],
  ["deepseek", "grpo"],
  ["simple_rl", "grpo"],

  // Interpretability.
  ["transformer", "mech-inter"],
  ["sae", "mech-inter"],
  ["mech-inter", "h-neurons"],

  // Representation learning.
  ["2024-08-18-ssl", "2024-11-25-ssl"],
  ["2024-08-18-ssl", "plato"],

  // Graphs and energy-based models.
  ["gcn", "temporal-gnn"],
  ["hopfield", "up"],
];

/** Resolve each ref to its post by full stem first, then by bare slug. */
export function resolvePrereqs<P>(
  posts: P[],
  stem: (p: P) => string,
  slug: (p: P) => string,
): [P, P][] {
  const lookup = new Map<string, P>();
  for (const p of posts) lookup.set(slug(p).toLowerCase(), p);
  for (const p of posts) lookup.set(stem(p).toLowerCase(), p);
  return prereqs
    .map(([a, b]) => [lookup.get(a.toLowerCase()), lookup.get(b.toLowerCase())])
    .filter((e): e is [P, P] => e[0] !== undefined && e[1] !== undefined);
}

/** Every ref must match exactly one post and the edges must stay acyclic. */
export function validatePrereqs(stems: string[], slugs: string[]) {
  const stemSet = new Set(stems.map((s) => s.toLowerCase()));
  const slugCount = new Map<string, number>();
  for (const s of slugs) {
    const k = s.toLowerCase();
    slugCount.set(k, (slugCount.get(k) ?? 0) + 1);
  }
  for (const edge of prereqs) {
    for (const ref of edge) {
      const k = ref.toLowerCase();
      if (stemSet.has(k)) continue;
      const n = slugCount.get(k) ?? 0;
      if (n === 0) {
        throw new Error(`prereq '${edge[0]}' -> '${edge[1]}': unknown post '${ref}'`);
      }
      if (n > 1) {
        throw new Error(
          `prereq '${edge[0]}' -> '${edge[1]}': slug '${ref}' is ambiguous, use the full YYYY-MM-DD-slug stem`,
        );
      }
    }
    if (edge[0].toLowerCase() === edge[1].toLowerCase()) {
      throw new Error(`prereq '${edge[0]}' points at itself`);
    }
  }

  // A cycle would make "Before reading this" chase its own tail.
  const out = new Map<string, string[]>();
  for (const [a, b] of prereqs) {
    const list = out.get(a.toLowerCase()) ?? [];
    list.push(b.toLowerCase());
    out.set(a.toLowerCase(), list);
  }
  const state = new Map<string, "open" | "done">();
  const visit = (node: string, path: string[]) => {
    if (state.get(node) === "done") return;
    if (state.get(node) === "open") {
      throw new Error(`prereq cycle: ${[...path, node].join(" -> ")}`);
    }
    state.set(node, "open");
    for (const next of out.get(node) ?? []) visit(next, [...path, node]);
    state.set(node, "done");
  };
  for (const node of out.keys()) visit(node, []);
}
