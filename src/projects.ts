// Project showcase data. Add a new entry here to make it appear on /projects.html.
// `image` is optional — cards without one render a generated monogram tile.

export type Project = {
  title: string;
  description: string;
  url: string;
  tags: string[];
  year?: number;
  image?: string;
};

export const projects: Project[] = [
  {
    title: "terror-reco",
    description:
      "Recommendation of horror movies based on AI and your current mood.",
    url: "https://github.com/mbottoni/terror-reco",
    tags: ["llm", "recsys"],
    year: 2024,
  },
  {
    title: "logit-graph",
    description: "Experiments with complex networks and random graphs.",
    url: "https://github.com/mbottoni/logit-graph",
    tags: ["graphs", "networks"],
    year: 2024,
  },
  {
    title: "ml-prod",
    description:
      "Machine learning in production: Kubernetes, AI workflows, and deployment strategies.",
    url: "https://github.com/mbottoni/ml-prod",
    tags: ["mlops", "kubernetes"],
    year: 2024,
  },
  {
    title: "flow-match",
    description:
      "Implementation and experiments with Flow Matching generative models.",
    url: "https://github.com/mbottoni/flow-match",
    tags: ["generative", "flow-matching"],
    year: 2025,
  },
  {
    title: "unet-fun",
    description:
      "Playing around with U-Net architectures and image segmentation tasks.",
    url: "https://github.com/mbottoni/unet-fun",
    tags: ["vision", "segmentation"],
    year: 2024,
  },
  {
    title: "kalman-transformer",
    description:
      "Exploring the intersection of Kalman Filters and Transformer architectures.",
    url: "https://github.com/mbottoni/kalman-transformer",
    tags: ["transformers", "kalman-filter"],
    year: 2024,
  },
  {
    title: "blur-filter-variations",
    description:
      "Implementation of various blur filters and image processing algorithms.",
    url: "https://github.com/mbottoni/blur_filter",
    tags: ["vision", "signal-processing"],
    year: 2023,
  },
  {
    title: "rl-rubiks-cube",
    description: "Project exploring RL algorithms with a Rubik's cube.",
    url: "https://github.com/mbottoni/reinforcement-learning-final-project",
    tags: ["reinforcement-learning"],
    year: 2023,
  },
  {
    title: "bert-trial",
    description: "BERT for classification project.",
    url: "https://github.com/mbottoni/ep2-nlp",
    tags: ["nlp", "bert"],
    year: 2023,
  },
  {
    title: "graph-corr-embedd",
    description:
      "Graph correlation embeddings and network analysis experiments.",
    url: "https://github.com/mbottoni/graph-corr-embedd",
    tags: ["graphs", "embeddings"],
    year: 2024,
  },
];
