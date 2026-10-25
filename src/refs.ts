// Bibliography for `[@key]` citations in posts. A post that cites anything
// gets a numbered "References" section; citing a key that is missing here
// fails the build.
//
// Keys are `<first-author-surname><year>` (add a suffix letter on collision).
// List every author in order; rendering abbreviates long lists to "et al.".
// Prefer an arXiv ID, then a DOI, then a plain URL.

export type Ref = {
  key: string;
  authors: string[];
  title: string;
  year: number;
  venue: string;
  arxiv?: string;
  doi?: string;
  url?: string;
};

export const refs: Ref[] = [
  {
    key: "frankle2019",
    authors: ["Jonathan Frankle", "Michael Carbin"],
    title:
      "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
    year: 2019,
    venue: "ICLR",
    arxiv: "1803.03635",
  },
  {
    key: "frankle2019stabilizing",
    authors: [
      "Jonathan Frankle",
      "Gintare Karolina Dziugaite",
      "Daniel M. Roy",
      "Michael Carbin",
    ],
    title: "Stabilizing the Lottery Ticket Hypothesis",
    year: 2019,
    venue: "arXiv preprint",
    arxiv: "1903.01611",
  },
  {
    key: "zhang2024relu2",
    authors: [
      "Zhengyan Zhang",
      "Yixin Song",
      "Guanghui Yu",
      "Xu Han",
      "Yankai Lin",
      "Chaojun Xiao",
      "Chenyang Song",
      "Zhiyuan Liu",
      "Zeyu Mi",
      "Maosong Sun",
    ],
    title: "ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs",
    year: 2024,
    venue: "arXiv preprint",
    arxiv: "2402.03804",
  },
  {
    key: "ferrando2025",
    authors: [
      "Javier Ferrando",
      "Oscar Obeso",
      "Senthooran Rajamanoharan",
      "Neel Nanda",
    ],
    title:
      "Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models",
    year: 2025,
    venue: "ICLR",
    arxiv: "2411.14257",
  },
  {
    key: "lindsey2025",
    // Abbreviated: the full list of 27 authors is on the page.
    authors: ["Jack Lindsey", "Wes Gurnee", "Emmanuel Ameisen", "Brian Chen"],
    title: "On the Biology of a Large Language Model",
    year: 2025,
    venue: "Transformer Circuits Thread",
    url: "https://transformer-circuits.pub/2025/attribution-graphs/biology.html",
  },
  {
    key: "gao2025",
    authors: [
      "Cheng Gao",
      "Huimin Chen",
      "Chaojun Xiao",
      "Zhiyi Chen",
      "Zhiyuan Liu",
      "Maosong Sun",
    ],
    title:
      "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs",
    year: 2025,
    venue: "arXiv preprint",
    arxiv: "2512.01797",
  },
  {
    key: "finzi2026",
    authors: [
      "Marc Finzi",
      "Shikai Qiu",
      "Yiding Jiang",
      "Pavel Izmailov",
      "J. Zico Kolter",
      "Andrew Gordon Wilson",
    ],
    title:
      "From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence",
    year: 2026,
    venue: "arXiv preprint",
    arxiv: "2601.03220",
  },
];

const lookup = new Map<string, Ref>();
for (const r of refs) {
  if (lookup.has(r.key)) throw new Error(`refs: duplicate key '${r.key}'`);
  lookup.set(r.key, r);
}

export function resolveRef(key: string): Ref | undefined {
  return lookup.get(key);
}

/** "A.", "A and B.", "A, B, and C.", or "A et al." beyond three authors. */
export function formatAuthors(authors: string[]): string {
  let s = `${authors[0]} et al.`;
  if (authors.length < 3) s = authors.join(" and ");
  if (authors.length == 3) s = `${authors[0]}, ${authors[1]}, and ${authors[2]}`;
  // Don't double the period after "et al." or a trailing initial.
  return s.endsWith(".") ? s : `${s}.`;
}

/** Where the reference's link points, with a short label for it. */
export function refLink(r: Ref): { href: string; label: string } | undefined {
  if (r.arxiv) {
    return { href: `https://arxiv.org/abs/${r.arxiv}`, label: `arXiv:${r.arxiv}` };
  }
  if (r.doi) return { href: `https://doi.org/${r.doi}`, label: `doi:${r.doi}` };
  if (r.url) return { href: r.url, label: new URL(r.url).hostname };
  return undefined;
}

/** Plain-text form, used for the hover tooltip on citation markers. */
export function formatRef(r: Ref): string {
  return `${formatAuthors(r.authors)} ${r.title}. ${r.venue}, ${r.year}.`;
}
