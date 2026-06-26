// Free-form tags per post slug, keyed by the lowercased slug from the filename
// (`YYYY-MM-DD-<slug>.dj`). A post with no entry simply shows no tags. Tag index
// and `/tags/<tag>.html` pages are generated only for tags that appear here.

const tagAssignments: Record<string, string[]> = {
  bertimbau: ["nlp", "bert", "transformers"],
  hopfield: ["hopfield", "associative-memory", "energy-based"],
  hopfield_from_scratch: ["hopfield", "from-scratch", "energy-based"],
  up: ["probability", "kolmogorov"],
  kalman: ["kalman-filter", "bayesian", "time-series"],
  transformer: ["transformers", "attention", "architecture"],
  decoding: ["llm", "inference", "sampling"],
  mandelbrot: ["fractals", "math", "visualization"],
  "llm-archs": ["llm", "architecture", "survey"],
  kolmogorov: ["kolmogorov", "information-theory"],
  simple_rl: ["reinforcement-learning", "from-scratch"],
  ctgan: ["gan", "tabular", "generative"],
  ssl: ["self-supervised", "representation-learning"],
  rag: ["rag", "llm", "retrieval"],
  diffusion: ["diffusion", "generative"],
  ddim_ddpm: ["diffusion", "generative", "sampling"],
  divergences: ["information-theory", "probability"],
  kan: ["kan", "architecture"],
  "llm-quant": ["quantization", "llm", "efficiency"],
  vaes: ["vae", "generative", "latent-variable"],
  iit: ["consciousness", "information-theory", "philosophy"],
  gcn: ["graph-neural-networks", "graphs"],
  "temporal-gnn": ["graph-neural-networks", "time-series", "graphs"],
  sae: ["interpretability", "sparse-autoencoders", "llm"],
  plato: ["philosophy", "representation-learning"],
  flow: ["flow-matching", "generative"],
  "timeseries-diffusion": ["diffusion", "time-series", "generative"],
  cfg: ["classifier-free-guidance", "diffusion", "generative"],
  prob: ["probability", "foundations"],
  deepseek: ["llm", "deepseek", "training"],
  grpo: ["reinforcement-learning", "llm", "rlhf"],
  "mech-inter": ["interpretability", "llm"],
  swarm: ["swarm", "agents", "reinforcement-learning"],
  "neural-collapse": ["deep-learning", "theory", "representation-learning"],
  moe: ["mixture-of-experts", "llm", "architecture"],
  epiplexity: ["information-theory", "probability"],
  "lottery-ticket": ["pruning", "lottery-ticket", "efficiency"],
  "h-neurons": ["interpretability", "llm"],
};

export function resolveTags(slug: string): string[] {
  return tagAssignments[slug.toLowerCase()] ?? [];
}

export function tagSlug(tag: string): string {
  return tag.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export function tagPath(tag: string): string {
  return `/tags/${tagSlug(tag)}.html`;
}
