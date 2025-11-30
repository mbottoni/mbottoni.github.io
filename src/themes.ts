export type ThemeKey =
  | "frontier"
  | "generative"
  | "graph_rl"
  | "foundations";

export type ThemeConfig = {
  key: ThemeKey;
  title: string;
  description: string;
  path: string;
};

const themeDefinitions: ThemeConfig[] = [
  {
    key: "frontier",
    title: "Frontier LLMs & Architectures",
    description:
      "Transformers, mixture-of-experts, interpretability, retrieval, self-supervision, and training techniques for large models.",
    path: "/themes/frontier.html",
  },
  {
    key: "generative",
    title: "Diffusion & Generative Modeling",
    description:
      "Diffusion processes, flows, GAN/CTGAN, CFG tricks, and other approaches to high-dimensional generation.",
    path: "/themes/generative.html",
  },
  {
    key: "graph_rl",
    title: "Graphs, Agents & RL",
    description:
      "Graph neural networks, swarm simulations, and reinforcement-learning flavored explorations.",
    path: "/themes/graph-rl.html",
  },
  {
    key: "foundations",
    title: "Math, Physics & Foundations",
    description:
      "Hopfield networks, Kalman filters, Kolmogorov ideas, probability, information theory, and philosophical musings.",
    path: "/themes/foundations.html",
  },
];

export const themes = themeDefinitions;

const themeMap = new Map<ThemeKey, ThemeConfig>(
  themeDefinitions.map((theme) => [theme.key, theme]),
);

const themeAssignments: Record<ThemeKey, string[]> = {
  frontier: [
    "bertimbau",
    "transformer",
    "decoding",
    "llm-archs",
    "rag",
    "llm-quant",
    "kan",
    "sae",
    "ssl",
    "deepseek",
    "grpo",
    "mech-inter",
    "neural-collapse",
    "moe",
  ],
  generative: [
    "ctgan",
    "diffusion",
    "ddim_ddpm",
    "flow",
    "timeseries-diffusion",
    "cfg",
    "vaes",
  ],
  graph_rl: [
    "simple_rl",
    "gcn",
    "temporal-gnn",
    "swarm",
  ],
  foundations: [
    "hopfield",
    "hopfield_from_scratch",
    "up",
    "kalman",
    "mandelbrot",
    "kolmogorov",
    "divergences",
    "iit",
    "plato",
    "prob",
  ],
};

const slugLookup = new Map<string, ThemeConfig>();
for (const [key, slugs] of Object.entries(themeAssignments)) {
  const theme = themeMap.get(key as ThemeKey);
  if (!theme) continue;
  for (const slug of slugs) {
    slugLookup.set(slug.toLowerCase(), theme);
  }
}

const defaultTheme = themeMap.get("foundations")!;

export function resolveTheme(slug: string): ThemeConfig {
  return slugLookup.get(slug.toLowerCase()) ?? defaultTheme;
}

