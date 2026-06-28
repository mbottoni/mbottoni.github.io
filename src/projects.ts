// Project showcase data. Add a new entry here to make it appear on /projects.html.
//
// `slug`  — used for the detail page URL (/projects/<slug>.html), the GIF asset
//           name, and the optional detail file content/projects/<slug>.dj.
//           When that .dj file exists, cards link to the detail page; otherwise
//           they link straight to `url`.
// `gif`   — optional demo animation (path under /assets), shown on the card and
//           the detail page. Drop the file at content/assets/projects/<slug>.gif.
// `gifCaption` — optional caption rendered under the GIF on the detail page.
// `image` — optional static thumbnail; cards with neither gif nor image render a
//           generated monogram tile.

export type Project = {
  title: string;
  slug: string;
  description: string;
  url: string;
  tags: string[];
  year?: number;
  gif?: string;
  gifCaption?: string;
  image?: string;
};

export const projects: Project[] = [
  {
    title: "fundamental-agents",
    slug: "fundamental-agents",
    description:
      "Multi-agent AI platform for fundamental stock analysis: an orchestrator runs five agents to turn a ticker into an investment report.",
    url: "https://github.com/mbottoni/fundamental-agents",
    tags: ["multi-agent", "fastapi", "nextjs"],
    year: 2025,
    gif: "/assets/projects/fundamental-agents.gif",
    gifCaption:
      "An orchestrator runs five agents in sequence — data gathering, financial metrics, news sentiment, valuation, synthesis — turning a ticker into an investment report.",
  },
  {
    title: "vae-playground",
    slug: "vae-playground",
    description:
      "A modular playground of five VAE variants (Vanilla, β-VAE, Conditional, VQ-VAE, WAE-MMD) with interactive marimo notebooks.",
    url: "https://github.com/mbottoni/vae-playground",
    tags: ["generative", "vae", "pytorch"],
    year: 2025,
    gif: "/assets/projects/vae-playground.gif",
    gifCaption:
      "A VAE's latent space organizes data into clusters; sampling a point z and decoding it generates a new image.",
  },
  {
    title: "terror-reco",
    slug: "terror-reco",
    description:
      "Horror movie recommendations from free-text mood, via semantic search.",
    url: "https://github.com/mbottoni/terror-reco",
    tags: ["semantic-search", "recsys", "fastapi"],
    year: 2024,
    gif: "/assets/projects/terror-reco.gif",
  },
  {
    title: "logit-graph",
    slug: "logit-graph",
    description:
      "Fit, simulate, and compare logit graph models against ER/WS/BA via spectral GIC. On PyPI.",
    url: "https://github.com/mbottoni/logit-graph",
    tags: ["random-graphs", "network-science", "pypi"],
    year: 2025,
    gif: "/assets/projects/logit-graph.gif",
    gifCaption:
      "Synthetic temporal logit graph: nodes arrive over time and attach preferentially to high-degree nodes, so hubs emerge as the network grows.",
  },
  {
    title: "ml-prod",
    slug: "ml-prod",
    description:
      "End-to-end ML in production: a model microservice, Docker, Kubernetes, and drift monitoring.",
    url: "https://github.com/mbottoni/ml-prod",
    tags: ["mlops", "docker", "kubernetes", "flask"],
    year: 2024,
    gif: "/assets/projects/ml-prod.gif",
    gifCaption:
      "The ML production lifecycle: data → train → Docker → Kubernetes → monitor, looping back to retrain when the monitor detects drift.",
  },
  {
    title: "flow-match",
    slug: "flow-match",
    description:
      "Flow Matching for generative modeling: a scikit-learn-like FlowMatcher transporting noise to data via learned ODE velocity fields.",
    url: "https://github.com/mbottoni/flow-match",
    tags: ["generative", "flow-matching", "pytorch"],
    year: 2025,
    gif: "/assets/projects/flow-match.gif",
    gifCaption:
      "Flow matching transports a Gaussian noise cloud into a two-moons distribution along the conditional flow path xₜ = (1−t)·x₀ + t·x₁.",
  },
  {
    title: "unet-fun",
    slug: "unet-fun",
    description:
      "A from-scratch U-Net for binary image segmentation (Carvana car masking) — encoder–decoder with skip connections.",
    url: "https://github.com/mbottoni/unet-fun",
    tags: ["vision", "segmentation", "unet"],
    year: 2024,
    gif: "/assets/projects/unet-fun.gif",
    gifCaption:
      "U-Net for segmentation: an image flows down the encoder, through the bottleneck, and up the decoder to a mask, with skip connections restoring spatial detail.",
  },
  {
    title: "kalman-transformer",
    slug: "kalman-transformer",
    description:
      "A Kalman filter and a Transformer as interchangeable expected-return estimators feeding a Markowitz backtest.",
    url: "https://github.com/mbottoni/kalman-transformer",
    tags: ["kalman-filter", "transformers", "portfolio-optimization"],
    year: 2024,
    gif: "/assets/projects/kalman-transformer.gif",
    gifCaption:
      "Expected-return estimation as state estimation: a Kalman filter and a Transformer both denoise noisy returns into the expected return that feeds Markowitz.",
  },
  {
    title: "rl-rubiks-cube",
    slug: "rl-rubiks-cube",
    description:
      "Self-taught Rubik's Cube solver — Autodidactic Iteration (value+policy net, no dataset) solving via beam search.",
    url: "https://github.com/mbottoni/reinforcement-learning-final-project",
    tags: ["reinforcement-learning", "deep-rl", "pytorch"],
    year: 2023,
    gif: "/assets/projects/rl-rubiks-cube.gif",
    gifCaption:
      "A scrambled cube net solving itself by replaying the scramble backwards — the same 'scramble, then walk it back' idea the agent trains on.",
  },
  {
    title: "bertimbau-probing",
    slug: "bertimbau-probing",
    description:
      "Fine-tuning BERTimbau on Portuguese reviews to probe whether its embeddings recover vowel density.",
    url: "https://github.com/mbottoni/ep2-nlp",
    tags: ["nlp", "bert", "portuguese", "probing"],
    year: 2023,
    gif: "/assets/projects/bertimbau-probing.gif",
    gifCaption:
      "Probing BERTimbau: a scan highlights vowels in each Portuguese review while the gauge shows the vowel density recovered from the embedding.",
  },
  {
    title: "graph-corr-embedd",
    slug: "graph-corr-embedd",
    description:
      "Measuring correlation between graphs via learned embeddings vs. classical graph-distance measures.",
    url: "https://github.com/dcuoliveira/graph-corr-embedd",
    tags: ["graphs", "embeddings", "network-science"],
    year: 2024,
    gif: "/assets/projects/graph-corr-embedd.gif",
    gifCaption:
      "Measuring graph correlation from embeddings: as the correlation ρ drops, graph B's edges fade and its embedding cloud pulls away from A's.",
  },
];
