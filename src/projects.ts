// Project showcase data. Add a new entry here to make it appear on /projects.html.
//
// `slug`  — used for the detail page URL (/projects/<slug>.html), the GIF asset
//           name, and the optional detail file content/projects/<slug>.dj.
//           When that .dj file exists, cards link to the detail page; otherwise
//           they link straight to `url`.
// `gif`   — optional demo animation (path under /assets), shown on the card and
//           the detail page. Drop the file at content/assets/projects/<slug>.gif.
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
  image?: string;
};

export const projects: Project[] = [
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
    image: "/assets/projects/logit-graph.png",
  },
  {
    title: "ml-prod",
    slug: "ml-prod",
    description:
      "Machine learning in production: Kubernetes, AI workflows, and deployment strategies.",
    url: "https://github.com/mbottoni/ml-prod",
    tags: ["mlops", "kubernetes"],
    year: 2024,
  },
  {
    title: "flow-match",
    slug: "flow-match",
    description:
      "Implementation and experiments with Flow Matching generative models.",
    url: "https://github.com/mbottoni/flow-match",
    tags: ["generative", "flow-matching"],
    year: 2025,
  },
  {
    title: "unet-fun",
    slug: "unet-fun",
    description:
      "Playing around with U-Net architectures and image segmentation tasks.",
    url: "https://github.com/mbottoni/unet-fun",
    tags: ["vision", "segmentation"],
    year: 2024,
  },
  {
    title: "kalman-transformer",
    slug: "kalman-transformer",
    description:
      "Exploring the intersection of Kalman Filters and Transformer architectures.",
    url: "https://github.com/mbottoni/kalman-transformer",
    tags: ["transformers", "kalman-filter"],
    year: 2024,
  },
  {
    title: "blur-filter-variations",
    slug: "blur-filter-variations",
    description:
      "Implementation of various blur filters and image processing algorithms.",
    url: "https://github.com/mbottoni/blur_filter",
    tags: ["vision", "signal-processing"],
    year: 2023,
  },
  {
    title: "rl-rubiks-cube",
    slug: "rl-rubiks-cube",
    description: "Project exploring RL algorithms with a Rubik's cube.",
    url: "https://github.com/mbottoni/reinforcement-learning-final-project",
    tags: ["reinforcement-learning"],
    year: 2023,
  },
  {
    title: "bert-trial",
    slug: "bert-trial",
    description: "BERT for classification project.",
    url: "https://github.com/mbottoni/ep2-nlp",
    tags: ["nlp", "bert"],
    year: 2023,
  },
  {
    title: "graph-corr-embedd",
    slug: "graph-corr-embedd",
    description:
      "Graph correlation embeddings and network analysis experiments.",
    url: "https://github.com/mbottoni/graph-corr-embedd",
    tags: ["graphs", "embeddings"],
    year: 2024,
  },
];
