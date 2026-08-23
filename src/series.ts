// Multi-part series. Each series lists its posts in reading order; a post
// page renders a "Part N of M" box linking to the other parts.
//
// Entries may be a bare slug (`hopfield`) or the full filename stem
// (`2024-11-25-ssl`) when two posts share a slug.

export type SeriesConfig = {
  key: string;
  title: string;
  description?: string;
  posts: string[];
};

export const series: SeriesConfig[] = [
  {
    key: "hopfield",
    title: "Hopfield networks",
    description: "From the energy-based picture to a from-scratch implementation.",
    posts: ["hopfield", "hopfield_from_scratch"],
  },
  {
    key: "diffusion",
    title: "Diffusion and flows",
    description:
      "Diffusion models, samplers, guidance, and their flow-matching cousins.",
    posts: ["diffusion", "ddim_ddpm", "cfg", "flow", "timeseries-diffusion"],
  },
  {
    key: "ssl",
    title: "Self-supervised learning",
    posts: ["2024-08-18-ssl", "2024-11-25-ssl"],
  },
  {
    key: "deepseek",
    title: "DeepSeek notes",
    posts: ["deepseek", "grpo"],
  },
  {
    key: "interp",
    title: "Mechanistic interpretability",
    posts: ["sae", "mech-inter", "h-neurons"],
  },
];

export type SeriesMembership = {
  series: SeriesConfig;
  index: number; // 0-based position in the series
};

const lookup = new Map<string, SeriesMembership>();
for (const s of series) {
  s.posts.forEach((ref, index) => {
    if (lookup.has(ref.toLowerCase())) {
      throw new Error(`post '${ref}' is listed in more than one series`);
    }
    lookup.set(ref.toLowerCase(), { series: s, index });
  });
}

/** Resolve by full stem first (`YYYY-MM-DD-slug`), then by bare slug. */
export function resolveSeries(
  stem: string,
  slug: string,
): SeriesMembership | undefined {
  return lookup.get(stem.toLowerCase()) ?? lookup.get(slug.toLowerCase());
}

/** Every ref in a series must match exactly one post; called after collect. */
export function validateSeries(stems: string[], slugs: string[]) {
  const stemSet = new Set(stems.map((s) => s.toLowerCase()));
  const slugCount = new Map<string, number>();
  for (const s of slugs) {
    const k = s.toLowerCase();
    slugCount.set(k, (slugCount.get(k) ?? 0) + 1);
  }
  for (const s of series) {
    for (const ref of s.posts) {
      const k = ref.toLowerCase();
      if (stemSet.has(k)) continue;
      const n = slugCount.get(k) ?? 0;
      if (n === 0) throw new Error(`series '${s.key}': unknown post '${ref}'`);
      if (n > 1) {
        throw new Error(
          `series '${s.key}': slug '${ref}' is ambiguous, use the full YYYY-MM-DD-slug stem`,
        );
      }
    }
  }
}
