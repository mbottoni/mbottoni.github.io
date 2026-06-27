// Short announcements shown on the homepage and on /news.html.
// Newest first. `date` is ISO yyyy-mm-dd. `html` may contain inline links.

export type NewsItem = {
  date: string;
  html: string;
};

export const news: NewsItem[] = [
  {
    date: "2026-01-27",
    html:
      `New post: <a href="/2026/01/27/h-neurons.html">Helpful Neurons in LLMs</a>.`,
  },
  {
    date: "2026-01-26",
    html:
      `New post: <a href="/2026/01/26/lottery-ticket.html">The Lottery Ticket Hypothesis</a>.`,
  },
  {
    date: "2026-01-25",
    html: `New post: <a href="/2026/01/25/epiplexity.html">Epiplexity</a>.`,
  },
  {
    date: "2025-08-04",
    html:
      `New post: <a href="/2025/08/04/moe.html">Mixture of Experts</a>, on sparse model scaling.`,
  },
];
