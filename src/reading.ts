// The /reading.html list: papers I keep coming back to, mixed across topics.
//
// Order is deliberate (roughly most-revisited first), so append or move entries
// by hand. `year` is the year of the first arXiv version.
//
// `posts`    — posts that discuss the paper: the bare slug, or the full
//              `YYYY-MM-DD-slug` stem when two posts share a slug.
// `projects` — slugs from src/projects.ts.
// The build fails if either points at something that does not exist.

export type ReadingItem = {
  arxiv: string;
  title: string;
  authors: string;
  year: number;
  note: string;
  posts?: string[];
  projects?: string[];
};

export type ElsewhereLink = { href: string; label: string; note: string };

export const reading: ReadingItem[] = [
  {
    arxiv: "math/0406077",
    title:
      "A Tutorial Introduction to the Minimum Description Length Principle",
    authors: "Grünwald",
    year: 2004,
    note:
      "Learning as compression: prefer the model that gives the shortest description of the data. Chapter 1 is a non-technical tour; chapter 2 makes it precise.",
    posts: ["kolmogorov", "epiplexity"],
  },
  {
    arxiv: "2107.04562",
    title: "The Bayesian Learning Rule",
    authors: "Khan & Rue",
    year: 2021,
    note:
      "Ridge regression, Newton's method, the Kalman filter, SGD, RMSprop and dropout all fall out of one natural-gradient Bayesian update.",
    posts: ["prob"],
  },
  {
    arxiv: "2407.09468",
    title:
      "Beyond Euclid: An Illustrated Guide to Modern Machine Learning with Geometric, Topological, and Algebraic Structures",
    authors: "Sanborn et al.",
    year: 2024,
    note:
      "An illustrated map of machine learning beyond flat vector spaces: data and models with geometric, topological and algebraic structure.",
  },
  {
    arxiv: "2406.08929",
    title: "Step-by-Step Diffusion: An Elementary Tutorial",
    authors: "Nakkiran et al.",
    year: 2024,
    note:
      "A first course on diffusion and flow matching that derives correct algorithms with as little machinery as possible.",
    posts: ["diffusion", "ddim_ddpm"],
  },
  {
    arxiv: "2310.03884",
    title: "Information Geometry for the Working Information Theorist",
    authors: "Mishra, Kumar & Wong",
    year: 2023,
    note:
      "Statistical manifolds, Fisher information and divergences from a geometric point of view, written for information theorists.",
    posts: ["divergences"],
    projects: ["fisher-rao-ml"],
  },
  {
    arxiv: "1512.06830",
    title:
      "Correlation Between Graphs with an Application to Brain Networks Analysis",
    authors: "Fujita et al.",
    year: 2015,
    note:
      "What does it mean for two graphs to be correlated? A definition through the parameters of the random graph models behind them, applied to brain networks.",
    projects: ["graph-corr-embedd"],
  },
  {
    arxiv: "2107.07511",
    title:
      "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification",
    authors: "Angelopoulos & Bates",
    year: 2021,
    note:
      "Wrap any trained model in prediction sets with distribution-free, finite-sample guarantees. Short and practical.",
  },
  {
    arxiv: "2202.00665",
    title: "Tutorial on Amortized Optimization",
    authors: "Amos",
    year: 2022,
    note:
      "Learning to predict the solutions of optimization problems: the idea behind the VAE encoder, and much of reinforcement learning.",
    posts: ["vaes"],
  },
  {
    arxiv: "2002.10689",
    title: "A Theory of Usable Information Under Computational Constraints",
    authors: "Xu et al.",
    year: 2020,
    note:
      "V-information: what a computationally bounded observer can actually extract. Unlike Shannon information, it can be created by computation.",
    posts: ["epiplexity"],
  },
  {
    arxiv: "2312.06937",
    title: "Can a Transformer Represent a Kalman Filter?",
    authors: "Goel & Bartlett",
    year: 2023,
    note:
      "Yes, up to a small error bounded uniformly in time: softmax attention can implement kernel smoothing, which tracks the Kalman filter.",
    posts: ["kalman"],
    projects: ["kalman-transformer"],
  },
  {
    arxiv: "2412.09563",
    title:
      "Does Representation Matter? Exploring Intermediate Layers in Large Language Models",
    authors: "Skean et al.",
    year: 2024,
    note:
      "Intermediate layers of LLMs often give better representations than the last one, for Transformers and state-space models alike.",
    posts: ["plato"],
  },
  {
    arxiv: "2309.07867",
    title: "Beta Diffusion",
    authors: "Zhou et al.",
    year: 2023,
    note:
      "Diffusion for data in bounded ranges, using multiplicative beta-distributed transitions instead of additive Gaussian noise.",
  },
  {
    arxiv: "2512.24880",
    title: "mHC: Manifold-Constrained Hyper-Connections",
    authors: "Xie et al.",
    year: 2025,
    note:
      "Hyper-connections widen the residual stream but break its identity mapping; mHC restores it by constraining the connections to a manifold.",
  },
  {
    arxiv: "2304.05366",
    title:
      "The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning",
    authors: "Goldblum et al.",
    year: 2023,
    note:
      "Real-world data has low Kolmogorov complexity and neural networks share that bias toward simplicity, so no-free-lunch theorems bite less than they seem.",
    posts: ["kolmogorov"],
  },
  {
    arxiv: "2407.00783",
    title: "Diffusion Models and Representation Learning: A Survey",
    authors: "Fuest et al.",
    year: 2024,
    note:
      "Both directions: what diffusion models learn as representations, and how representations are used to improve diffusion.",
    posts: ["diffusion"],
  },
  {
    arxiv: "2002.02405",
    title: "How Good is the Bayes Posterior in Deep Neural Networks Really?",
    authors: "Wenzel et al.",
    year: 2020,
    note:
      "The cold posterior effect: in deep nets, a deliberately sharpened posterior predicts better than the actual Bayes posterior.",
    posts: ["prob"],
  },
  {
    arxiv: "1601.00670",
    title: "Variational Inference: A Review for Statisticians",
    authors: "Blei, Kucukelbir & McAuliffe",
    year: 2016,
    note:
      "The standard introduction to variational inference: posit a family, minimize the KL to the posterior, and scale it with stochastic optimization.",
    posts: ["vaes"],
  },
  {
    arxiv: "1706.03762",
    title: "Attention Is All You Need",
    authors: "Vaswani et al.",
    year: 2017,
    note: "The Transformer: attention only, no recurrence or convolutions.",
    posts: ["transformer"],
  },
  {
    arxiv: "2102.11224",
    title:
      "Spectral Density of Random Graphs: Convergence Properties and Application in Model Fitting",
    authors: "Santos, Fujita & Matias",
    year: 2021,
    note:
      "The spectral density of a graph's adjacency matrix as a model-agnostic tool to fit and select random graph models.",
    projects: ["logit-graph"],
  },
  {
    arxiv: "2304.00803",
    title: "A Tutorial Introduction to Reinforcement Learning",
    authors: "Vidyasagar",
    year: 2023,
    note:
      "A compact survey of reinforcement learning that treats TD learning and Q-learning as stochastic approximation.",
    posts: ["simple_rl"],
  },
  {
    arxiv: "1911.01547",
    title: "On the Measure of Intelligence",
    authors: "Chollet",
    year: 2019,
    note:
      "Intelligence as skill-acquisition efficiency rather than skill, with the ARC benchmark to measure it.",
  },
  {
    arxiv: "2206.04041",
    title:
      "Neural Collapse: A Review on Modelling Principles and Generalization",
    authors: "Kothapalli",
    year: 2022,
    note:
      "Late in training, last-layer features collapse to their class means, which form a simplex equiangular tight frame. A review of why, and what it implies.",
    posts: ["neural-collapse"],
  },
  {
    arxiv: "2308.07037",
    title: "Bayesian Flow Networks",
    authors: "Graves et al.",
    year: 2023,
    note:
      "Instead of denoising samples, update the parameters of simple distributions by Bayesian inference. No forward process, and discrete data works natively.",
  },
  {
    arxiv: "2502.09992",
    title: "Large Language Diffusion Models",
    authors: "Nie et al.",
    year: 2025,
    note:
      "LLaDA: a language model trained from scratch as a masked diffusion model rather than autoregressively, and competitive at 8B scale.",
  },
  {
    arxiv: "2602.04770",
    title: "Generative Modeling via Drifting",
    authors: "Deng et al.",
    year: 2026,
    note:
      "Drifting models move the generated distribution during training instead of at inference time, so sampling takes a single step.",
  },
  {
    arxiv: "1609.01214",
    title:
      "Spectral Entropies as Information-Theoretic Tools for Complex Network Comparison",
    authors: "De Domenico & Biamonte",
    year: 2016,
    note:
      "An entropy for networks from their spectrum, borrowed from quantum statistical mechanics, which yields a Jensen-Shannon distance between networks.",
  },
  {
    arxiv: "1607.00653",
    title: "node2vec: Scalable Feature Learning for Networks",
    authors: "Grover & Leskovec",
    year: 2016,
    note:
      "Node embeddings from biased random walks that interpolate between breadth-first and depth-first exploration.",
  },
  {
    arxiv: "2111.09266",
    title: "GFlowNet Foundations",
    authors: "Bengio et al.",
    year: 2021,
    note:
      "Learn to sample objects like sets and graphs with probability proportional to a reward, amortizing work that MCMC would otherwise do.",
  },
  {
    arxiv: "1511.05121",
    title: "Deep Kalman Filters",
    authors: "Krishnan, Shalit & Sontag",
    year: 2015,
    note:
      "Kalman filters with neural-network transitions and emissions, learned with the variational machinery of VAEs.",
    posts: ["kalman", "vaes"],
  },
  {
    arxiv: "2505.09343",
    title:
      "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures",
    authors: "Zhao et al.",
    year: 2025,
    note:
      "How DeepSeek-V3 was co-designed with its hardware: Multi-head Latent Attention, mixture of experts, FP8 training and the interconnect limits behind them.",
    posts: ["deepseek", "moe"],
  },
  {
    arxiv: "2210.02747",
    title: "Flow Matching for Generative Modeling",
    authors: "Lipman et al.",
    year: 2022,
    note:
      "Train continuous normalizing flows without simulation, by regressing vector fields along fixed conditional paths; diffusion paths are a special case.",
    posts: ["flow"],
    projects: ["flow-match"],
  },
  {
    arxiv: "1803.03635",
    title:
      "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
    authors: "Frankle & Carbin",
    year: 2018,
    note:
      "Dense networks contain sparse subnetworks that, reset to their original initialization, train to the same accuracy.",
    posts: ["lottery-ticket"],
  },
  {
    arxiv: "1710.10903",
    title: "Graph Attention Networks",
    authors: "Veličković et al.",
    year: 2017,
    note:
      "Message passing where each node learns how much weight to give each neighbor, via masked self-attention.",
    posts: ["gcn"],
  },
  {
    arxiv: "2404.19756",
    title: "KAN: Kolmogorov-Arnold Networks",
    authors: "Liu et al.",
    year: 2024,
    note:
      "Learnable activation functions on the edges instead of fixed ones on the nodes, inspired by the Kolmogorov-Arnold representation theorem.",
    posts: ["kan"],
  },
];

// Non-paper links kept from the old /links.html page.
export const elsewhere: ElsewhereLink[] = [
  {
    href: "https://terrytao.wordpress.com/",
    label: "Terence Tao's blog",
    note: "mathematics, from expository to research",
  },
  {
    href: "https://qchu.wordpress.com/reading-recommendations/",
    label: "Qiaochu Yuan's reading recommendations",
    note: "what to read in mathematics, and in what order",
  },
  {
    href: "https://dec41.user.srcf.net/",
    label: "Dexter Chua's notes",
    note: "lecture notes from the Cambridge Mathematical Tripos",
  },
  {
    href: "https://complexsystemstheory.net/references-2/",
    label: "Complex Systems Theory: further reading",
    note: "references on complex systems",
  },
  {
    href: "https://github.com/dppalomar?tab=repositories",
    label: "Daniel Palomar's repositories",
    note: "portfolio optimization and financial signal processing",
  },
];
