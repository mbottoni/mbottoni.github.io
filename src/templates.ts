// deno-lint-ignore-file no-explicit-any
import { Post } from "./main.ts";
import { themes, type ThemeConfig } from "./themes.ts";

const site_url = "https://mbottoni.github.io";

export const base = (
  { content, src, title, path, description, extra_css }: {
    content: HtmlString;
    src: string;
    title: string;
    description: string;
    path: string;
    extra_css?: string;
  },
): HtmlString =>
  html`
<!DOCTYPE html>
<html lang='en-US'>
<head>
  <meta charset='utf-8'>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title}</title>
  <meta name="description" content="${description}">
  <link rel="icon" href="/favicon.png" type="image/png">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="canonical" href="${site_url}${path}">
  <link rel="alternate" type="application/rss+xml" title="mbottoni" href="${site_url}/feed.xml">
  <style>
  @font-face {
    font-family: 'Open Sans'; src: url('/css/OpenSans-300-Normal.woff2') format('woff2');
    font-weight: 300; font-style: normal;
  }
  @font-face {
    font-family: 'JetBrains Mono'; src: url('/css/JetBrainsMono-400-Normal.woff2') format('woff2');
    font-weight: 400; font-style: normal;
  }
  @font-face {
    font-family: 'JetBrains Mono'; src: url('/css/JetBrainsMono-700-Normal.woff2') format('woff2');
    font-weight: 700; font-style: normal;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-400-Normal.woff2') format('woff2');
    font-weight: 400; font-style: normal;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-400-Italic.woff2') format('woff2');
    font-weight: 400; font-style: italic;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-700-Normal.woff2') format('woff2');
    font-weight: 700; font-style: normal;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-700-Italic.woff2') format('woff2');
    font-weight: 700; font-style: italic;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; margin-block-start: 0; margin-block-end: 0; }

  body {
    max-width: 80ch;
    padding: 2ch;
    margin-left: auto;
    margin-right: auto;
  }

  header { margin-bottom: 2rem; }
  header > nav { display: flex; column-gap: 2ch; align-items: baseline; flex-wrap: wrap; }
  header a { font-style: normal; color: rgba(0, 0, 0, .8); text-decoration: none; }
  header a:hover { color: rgba(0, 0, 0, .8); text-decoration: underline; }
  header .title { font-size: 1.25em; flex-grow: 2; }

  footer { margin-top: 2rem; }
  footer > p { display: flex; column-gap: 2ch; justify-content: center; flex-wrap: wrap; }
  footer a { color: rgba(0, 0, 0, .8); text-decoration: none; white-space: nowrap; }
  footer i { vertical-align: middle; color: rgba(0, 0, 0, .8) }

  .theme-grid { display: grid; gap: 2rem; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); }
  .theme-card { border: 1px solid rgba(0,0,0,.12); border-radius: 1rem; padding: 1.5rem; display: flex; flex-direction: column; gap: .75rem; }
  .theme-card h2 { font-size: 1.35em; }
  .theme-card h2 a { text-decoration: none; color: #b32f1c; }
  .theme-card h2 a:hover { text-decoration: underline; }
  .theme-card p { color: rgba(0,0,0,.75); line-height: 1.5; }
  .theme-meta { font-size: .85em; color: rgba(0,0,0,.6); }

  .theme-page header h1 { font-size: 2em; margin-bottom: .5rem; }
  .theme-page header p { color: rgba(0,0,0,.65); line-height: 1.5; }
  .theme-page { display: flex; flex-direction: column; gap: 1.5rem; }

  .post-list { list-style: none; display: flex; flex-direction: column; gap: 1.5rem; padding: 0; }
  .post-card { display: grid; gap: 1rem; grid-template-columns: minmax(0, 1fr); }
  .post-card__media { display: none; }
  .post-card__body h3 { font-size: 1.1em; margin-bottom: .35rem; }
  .post-card__body p { color: rgba(0, 0, 0, .7); }
  .post-card__media a { display: block; border-radius: .75rem; overflow: hidden; }
  .post-card__media img { width: 100%; height: 100%; object-fit: cover; display: block; }
  @media (min-width: 720px) {
    .post-card { grid-template-columns: 260px 1fr; align-items: center; }
    .post-card__media { display: block; min-height: 160px; }
    .post-card__media:empty { display: block; }
  }

  .theme-pill, .theme-banner a { display: inline-flex; align-items: center; gap: .4rem; font-size: .8em; text-transform: uppercase; letter-spacing: .08em; border: 1px solid rgba(0, 0, 0, .15); border-radius: 999px; padding: .25rem .9rem; text-decoration: none; color: rgba(0, 0, 0, .7); }
  .theme-pill svg, .theme-banner svg { width: .75em; height: .75em; }
  .theme-banner { margin-bottom: 1rem; }
  .theme-banner span { font-size: .8em; color: rgba(0, 0, 0, .6); margin-right: .5rem; }

  </style>

  <link rel="stylesheet" href="/css/main.css">
  ${extra_css ? html`<link rel="stylesheet" href="/css/${extra_css}">` : ""}
</head>

<body>
  <header>
    <nav>
      <a class="title" href="/">mbottoni</a>
      <a href="/about.html">About</a>
      <a href="/resume.html">Resume</a>
      <a href="/links.html">Links</a>
    </nav>
  </header>

  <main>
  ${content}
  </main>

  <footer class="site-footer">
    <p>
      <a href="https://github.com/mbottoni/mbottoni.github.io/edit/master${src}">
        <svg class="icon"><use href="/assets/icons.svg#edit"/></svg>
        Fix typo
      </a>
      <a href="/feed.xml">
        <svg class="icon"><use href="/assets/icons.svg#rss"/></svg>
        Subscribe
      </a>
      <a href="mailto:maruanbakriottoni@gmail.com">
        <svg class="icon"><use href="/assets/icons.svg#email"/></svg>
        Get in touch
      </a>
      <a href="https://github.com/mbottoni">
        <svg class="icon"><use href="/assets/icons.svg#github"/></svg>
        mbottoni
      </a>
    </p>
  </footer>
</body>

</html>
`;

const blurb = "Yet another programming blog by Maruan Bakri Ottoni aka mbottoni.";

export type ThemeGroup = {
  theme: ThemeConfig;
  posts: Post[];
};

export function page(name: string, content: HtmlString) {
  return base({
    path: `/${name}`,
    title: "mbottoni",
    description: blurb,
    src: `/content/${name}.dj`,
    extra_css: name === "resume" ? "resume.css" : undefined,
    content,
  });
}

export const post_list = (groups: ThemeGroup[]): HtmlString => {
  const cards = groups.map((group) => {
    const latest = group.posts[0];
    return html`
      <article class="theme-card">
        <div>
          <h2><a href="${group.theme.path}">${group.theme.title}</a></h2>
          <p>${group.theme.description}</p>
        </div>
        <p class="theme-meta">
          ${group.posts.length} post${group.posts.length === 1 ? "" : "s"}
          · Latest: ${time(latest.date)}
        </p>
      </article>
    `;
  });

  return base({
    path: "",
    title: "mbottoni",
    description: blurb,
    src: "/src/templates.ts",
    content: html`
      <section class="theme-grid">
        ${cards}
      </section>
    `,
  });
};

export const theme_page = (theme: ThemeConfig, posts: Post[]): HtmlString => {
  const items = posts.map((post) =>
    html`
      <li class="post-card">
        <div class="post-card__media">
          ${
        post.image
          ? html`<a href="${post.path}"><img src="${post.image}" alt="${post.title} preview"></a>`
          : ""
      }
        </div>
        <div class="post-card__body">
          <h3>${time(post.date)} · <a href="${post.path}">${post.title}</a></h3>
          <p>${post.summary}</p>
        </div>
      </li>
    `
  );

  return base({
    path: theme.path,
    title: `${theme.title} — mbottoni`,
    description: theme.description,
    src: "/src/templates.ts",
    content: html`
      <section class="theme-page">
        <header class="theme-section">
          <h1>${theme.title}</h1>
          <p>${theme.description}</p>
        </header>
        <ul class="post-list">
          ${items}
        </ul>
      </section>
    `,
  });
};

export function post(post: Post, spellcheck: boolean): HtmlString {
  return base({
    src: post.src,
    title: post.title,
    description: post.summary,
    path: post.path,
    content: html`
      <div class="theme-banner">
        <span>Filed under</span>
        <a class="theme-pill" href="${post.theme.path}">
          ${post.theme.title}
        </a>
      </div>
      <article ${
      spellcheck ? 'contentEditable="true"' : ""
    }>\n${post.content}</article>
    `,
  });
}

export function time(date: Date): HtmlString {
  const human = date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
  const machine = yyyy_mm_dd(date);
  return html`<time datetime="${machine}">${human}</time>`;
}

function yyyy_mm_dd(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export const feed = (posts: Post[]): HtmlString => {
  const entries = posts.slice(0, 10).map(feed_entry);

  return html`<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<link href="${site_url}/feed.xml" rel="self" type="application/atom+xml"/>
<link href="${site_url}" rel="alternate" type="text/html"/>
<updated>${new Date().toISOString()}</updated>
<id>${site_url}/feed.xml</id>
<title type="html">mbottoni</title>
<subtitle>Yet another programming blog by Maruan Bakri Ottoni aka mbottoni.</subtitle>
<author><name>Alex Kladov</name></author>
${entries}
</feed>
`;
};

export const feed_entry = (post: Post): HtmlString => {
  return html`
<entry>
<title type="text">${post.title}</title>
<link href="${site_url}${post.path}" rel="alternate" type="text/html" title="${post.title}" />
<published>${yyyy_mm_dd(post.date)}T00:00:00+00:00</published>
<updated>${yyyy_mm_dd(post.date)}T00:00:00+00:00</updated>
<id>${site_url}${post.path.replace(".html", "")}</id>
<author><name>Alex Kladov</name></author>
<summary type="html"><![CDATA[${post.summary}]]></summary>
<content type="html" xml:base="${site_url}${post.path}"><![CDATA[${post.content}]]></content>
</entry>
`;
};

export function html(
  strings: ArrayLike<string>,
  ...values: any[]
): HtmlString {
  function content(value: any): string[] {
    if (value === undefined) return [];
    if (value instanceof HtmlString) return [value.value];
    if (Array.isArray(value)) return value.flatMap(content);
    return [escapeHtml(value)];
  }
  return new HtmlString(
    String.raw({ raw: strings }, ...values.map((it) => content(it).join(""))),
  );
}

export class HtmlString {
  constructor(public value: string) {
  }
  push(other: HtmlString) {
    this.value = `${this.value}\n${other.value}`;
  }
}

function escapeHtml(data: any): string {
  return `${data}`
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
