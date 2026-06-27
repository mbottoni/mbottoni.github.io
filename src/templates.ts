// deno-lint-ignore-file no-explicit-any
import { Post } from "./main.ts";
import { type ThemeConfig } from "./themes.ts";
import { type Project } from "./projects.ts";
import { type NewsItem } from "./news.ts";
import { tagPath } from "./tags.ts";

const site_url = "https://mbottoni.github.io";
const author = "Maruan Bakri Ottoni";
const avatar = "https://avatars.githubusercontent.com/u/37229305?v=4";

const socials = [
  { icon: "github", href: "https://github.com/mbottoni", label: "GitHub" },
  { icon: "email", href: "mailto:maruanbakriottoni@gmail.com", label: "Email" },
  { icon: "rss", href: "/feed.xml", label: "RSS" },
];

const nav_links = [
  { href: "/", label: "Blog" },
  { href: "/projects.html", label: "Projects" },
  { href: "/news.html", label: "News" },
  { href: "/archive.html", label: "Archive" },
  { href: "/about.html", label: "About" },
  { href: "/resume.html", label: "Resume" },
  { href: "/links.html", label: "Links" },
];

const moon_svg =
  `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>`;
const search_svg =
  `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" d="M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zm5 12 4 4"/></svg>`;

export const base = (
  { content, src, title, path, description, extra_css, body_end, image }: {
    content: HtmlString;
    src: string;
    title: string;
    description: string;
    path: string;
    extra_css?: string;
    body_end?: HtmlString;
    image?: string;
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
  <meta name="author" content="${author}">
  <meta property="og:type" content="website">
  <meta property="og:title" content="${title}">
  <meta property="og:description" content="${description}">
  <meta property="og:url" content="${site_url}${path}">
  <meta property="og:image" content="${image ?? avatar}">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="${title}">
  <meta name="twitter:description" content="${description}">
  <meta name="twitter:image" content="${image ?? avatar}">
  <link rel="icon" href="/favicon.png" type="image/png">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="canonical" href="${site_url}${path}">
  <link rel="alternate" type="application/rss+xml" title="mbottoni" href="${site_url}/feed.xml">
  <script>
    (function () {
      try {
        var t = localStorage.getItem("theme");
        if (!t) t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", t);
      } catch (e) {}
    })();
  </script>
  <style>
  @font-face {
    font-family: 'Open Sans'; src: url('/css/OpenSans-300-Normal.woff2') format('woff2');
    font-weight: 300; font-style: normal; font-display: swap;
  }
  @font-face {
    font-family: 'JetBrains Mono'; src: url('/css/JetBrainsMono-400-Normal.woff2') format('woff2');
    font-weight: 400; font-style: normal; font-display: swap;
  }
  @font-face {
    font-family: 'JetBrains Mono'; src: url('/css/JetBrainsMono-700-Normal.woff2') format('woff2');
    font-weight: 700; font-style: normal; font-display: swap;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-400-Normal.woff2') format('woff2');
    font-weight: 400; font-style: normal; font-display: swap;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-400-Italic.woff2') format('woff2');
    font-weight: 400; font-style: italic; font-display: swap;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-700-Normal.woff2') format('woff2');
    font-weight: 700; font-style: normal; font-display: swap;
  }
  @font-face {
    font-family: 'EB Garamond'; src: url('/css/EBGaramond-700-Italic.woff2') format('woff2');
    font-weight: 700; font-style: italic; font-display: swap;
  }
  </style>

  <link rel="stylesheet" href="/css/main.css">
  ${extra_css ? html`<link rel="stylesheet" href="/css/${extra_css}">` : ""}
  <link rel="stylesheet" href="/css/katex/katex.min.css">
</head>

<body>
  <header class="site-header">
    <nav class="navbar">
      <a class="brand" href="/">mbottoni</a>
      <input type="checkbox" id="nav-toggle" class="nav-toggle" hidden>
      <label for="nav-toggle" class="nav-burger" aria-label="Toggle navigation">
        <span></span><span></span><span></span>
      </label>
      <div class="nav-links">
        ${
    nav_links.map((l) =>
      html`<a href="${l.href}"${
        l.href === path || (l.href === "/" && path === "")
          ? html` class="active"`
          : ""
      }>${l.label}</a>`
    )
  }
        <a class="nav-icon" href="/search.html" aria-label="Search">${
    new HtmlString(search_svg)
  }</a>
        <button class="nav-icon theme-toggle" type="button" aria-label="Toggle dark mode">${
    new HtmlString(moon_svg)
  }</button>
      </div>
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
  <script defer src="/css/katex/katex.min.js"></script>
  <script defer src="/css/katex/contrib/auto-render.min.js"></script>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    mermaid.initialize({ startOnLoad: true, theme: dark ? 'dark' : 'default' });
  </script>
  <script>
    (function () {
      var btn = document.querySelector(".theme-toggle");
      if (btn) {
        btn.addEventListener("click", function () {
          var next = document.documentElement.getAttribute("data-theme") === "dark"
            ? "light" : "dark";
          document.documentElement.setAttribute("data-theme", next);
          try { localStorage.setItem("theme", next); } catch (e) {}
        });
      }
    })();

    function unwrapPlainSpans(root) {
      for (const span of root.querySelectorAll("span")) {
        if (!(span instanceof HTMLElement)) continue;
        if (span.attributes.length === 0 && span.childElementCount === 0) {
          span.replaceWith(document.createTextNode(span.textContent || ""));
        }
      }
      root.normalize();
    }

    function addCopyButtons() {
      document.querySelectorAll('figure.code-block').forEach(block => {
        if (block.querySelector('.copy-button')) return;
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', () => {
          const code = block.querySelector('code')?.innerText || '';
          navigator.clipboard.writeText(code).then(() => {
            button.textContent = 'Copied!';
            setTimeout(() => button.textContent = 'Copy', 2000);
          });
        });
        block.appendChild(button);
      });
    }

    document.addEventListener("DOMContentLoaded", () => {
      unwrapPlainSpans(document.body);
      addCopyButtons();
      let attempts = 0;
      const maxAttempts = 40;
      const renderMath = () => {
        if (typeof renderMathInElement === "function") {
          try {
            renderMathInElement(document.body, {
              delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "$", right: "$", display: false },
                { left: "\\\\(", right: "\\\\)", display: false },
                { left: "\\\\[", right: "\\\\]", display: true },
              ],
              throwOnError: false,
              ignoredTags: [
                "script", "noscript", "style", "textarea",
                "pre", "code", "option"
              ],
              ignoredClasses: ["code-block", "hljs"],
            });
          } catch (e) {
            console.error("Math rendering error:", e);
          }
        } else if (attempts < maxAttempts) {
          attempts += 1;
          setTimeout(renderMath, 75);
        }
      };
      renderMath();
    });
  </script>
  ${body_end ?? ""}
</body>

</html>
`;

const blurb =
  "Notes on machine learning, math, and elegant ideas by Maruan Bakri Ottoni.";

export type ThemeGroup = {
  theme: ThemeConfig;
  posts: Post[];
};

export function page(name: string, content: HtmlString) {
  return base({
    path: `/${name}.html`,
    title: "mbottoni",
    description: blurb,
    src: `/content/${name}.dj`,
    extra_css: name === "resume" ? "resume.css" : undefined,
    content,
  });
}

function tag_pills(tags: string[]): HtmlString {
  if (tags.length === 0) return new HtmlString("");
  return html`<div class="tag-row">${
    tags.map((t) => html`<a class="tag" href="${tagPath(t)}">#${t}</a>`)
  }</div>`;
}

export const post_list = (
  groups: ThemeGroup[],
  posts: Post[],
  news: NewsItem[],
): HtmlString => {
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

  const recent = posts.slice(0, 5).map((post) =>
    html`<li><a href="${post.path}">${post.title}</a> <span class="muted">${
      time(post.date)
    }</span></li>`
  );

  return base({
    path: "",
    title: "mbottoni",
    description: blurb,
    src: "/src/templates.ts",
    content: html`
      <section class="hero">
        <img class="hero-ava" src="${avatar}" alt="${author}" width="120" height="120">
        <div class="hero-body">
          <h1>Maruan Bakri Ottoni</h1>
          <p>Data scientist who loves math and elegant ideas. I write about machine
          learning, generative models, and the foundations underneath them.</p>
          <p class="hero-socials">${
      socials.map((s) =>
        html`<a href="${s.href}" aria-label="${s.label}"><svg class="icon"><use href="/assets/icons.svg#${s.icon}"/></svg></a>`
      )
    }</div>
      </section>

      ${news_block(news)}

      <section class="home-section">
        <h2>Topics</h2>
        <div class="theme-grid">
          ${cards}
        </div>
      </section>

      <section class="home-section">
        <h2>Recent posts</h2>
        <ul class="recent-list">
          ${recent}
        </ul>
        <p><a href="/archive.html">See all posts →</a></p>
      </section>
    `,
  });
};

function news_block(news: NewsItem[]): HtmlString {
  if (news.length === 0) return new HtmlString("");
  const items = news.slice(0, 4).map((n) =>
    html`<li><span class="news-date">${news_date(n.date)}</span><span>${
      new HtmlString(n.html)
    }</span></li>`
  );
  return html`
    <section class="home-section">
      <h2>News</h2>
      <ul class="news-list">${items}</ul>
      <p><a href="/news.html">All news →</a></p>
    </section>
  `;
}

export const news_page = (news: NewsItem[]): HtmlString => {
  const items = news.map((n) =>
    html`<li><span class="news-date">${news_date(n.date)}</span><span>${
      new HtmlString(n.html)
    }</span></li>`
  );
  return base({
    path: "/news.html",
    title: "News — mbottoni",
    description: "Announcements and updates.",
    src: "/src/news.ts",
    content: html`
      <section class="theme-page">
        <header><h1>News</h1></header>
        <ul class="news-list">${items}</ul>
      </section>
    `,
  });
};

export const projects_page = (projects: Project[]): HtmlString => {
  const cards = projects.map((p) => {
    const initials = p.title.slice(0, 2).toUpperCase();
    const media = p.image
      ? html`<img src="${p.image}" alt="${p.title}">`
      : html`<span class="project-monogram">${initials}</span>`;
    return html`
      <article class="project-card">
        <a class="project-media" href="${p.url}">${media}</a>
        <div class="project-body">
          <h3><a href="${p.url}">${p.title}</a>${
      p.year ? html` <span class="muted">${p.year}</span>` : ""
    }</h3>
          <p>${p.description}</p>
          ${tag_pills(p.tags)}
        </div>
      </article>
    `;
  });
  return base({
    path: "/projects.html",
    title: "Projects — mbottoni",
    description: "Open source projects and experiments.",
    src: "/src/projects.ts",
    content: html`
      <section class="theme-page">
        <header>
          <h1>Projects</h1>
          <p>A sample of my open source projects and experiments. More on
          <a href="https://github.com/mbottoni">GitHub</a>.</p>
        </header>
        <div class="project-grid">${cards}</div>
      </section>
    `,
  });
};

export const archive_page = (posts: Post[]): HtmlString => {
  const byYear = new Map<number, Post[]>();
  for (const post of posts) {
    const list = byYear.get(post.year) ?? [];
    list.push(post);
    byYear.set(post.year, list);
  }
  const years = [...byYear.keys()].sort((a, b) => b - a);
  const sections = years.map((year) =>
    html`
      <section class="archive-year">
        <h2>${year}</h2>
        <ul class="archive-list">
          ${
      byYear.get(year)!.map((post) =>
        html`<li>
              <span class="muted">${time(post.date)}</span>
              <a href="${post.path}">${post.title}</a>
              <a class="archive-theme" href="${post.theme.path}">${post.theme.title}</a>
            </li>`
      )
    }
        </ul>
      </section>
    `
  );
  return base({
    path: "/archive.html",
    title: "Archive — mbottoni",
    description: "All posts, by year.",
    src: "/src/templates.ts",
    content: html`
      <section class="theme-page">
        <header>
          <h1>Archive</h1>
          <p>${posts.length} posts. <a href="/tags.html">Browse by tag →</a></p>
        </header>
        ${sections}
      </section>
    `,
  });
};

export const tags_index_page = (
  tags: { tag: string; count: number }[],
): HtmlString => {
  const sorted = [...tags].sort((a, b) =>
    b.count - a.count || a.tag.localeCompare(b.tag)
  );
  return base({
    path: "/tags.html",
    title: "Tags — mbottoni",
    description: "Browse posts by tag.",
    src: "/src/tags.ts",
    content: html`
      <section class="theme-page">
        <header><h1>Tags</h1></header>
        <div class="tag-cloud">
          ${
      sorted.map((t) =>
        html`<a class="tag" href="${tagPath(t.tag)}">#${t.tag} <span class="muted">${t.count}</span></a>`
      )
    }
        </div>
      </section>
    `,
  });
};

export const tag_page = (tag: string, posts: Post[]): HtmlString => {
  const items = posts.map((post) =>
    html`<li class="post-card">
      <div class="post-card__body">
        <h3>${time(post.date)} · <a href="${post.path}">${post.title}</a></h3>
        <p>${post.summary}</p>
      </div>
    </li>`
  );
  return base({
    path: tagPath(tag),
    title: `#${tag} — mbottoni`,
    description: `Posts tagged ${tag}.`,
    src: "/src/tags.ts",
    content: html`
      <section class="theme-page">
        <header>
          <h1>#${tag}</h1>
          <p>${posts.length} post${posts.length === 1 ? "" : "s"} · <a href="/tags.html">all tags</a></p>
        </header>
        <ul class="post-list">${items}</ul>
      </section>
    `,
  });
};

export const search_page = (): HtmlString => {
  return base({
    path: "/search.html",
    title: "Search — mbottoni",
    description: "Search the blog.",
    src: "/src/templates.ts",
    content: html`
      <section class="theme-page">
        <header><h1>Search</h1></header>
        <input id="search-input" class="search-input" type="search"
          placeholder="Search posts…" autocomplete="off" autofocus>
        <ul id="search-results" class="post-list"></ul>
        <p id="search-empty" class="muted" hidden>No matches.</p>
      </section>
    `,
    body_end: new HtmlString(`<script>
(function () {
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  var empty = document.getElementById("search-empty");
  var data = [];
  fetch("/search.json").then(function (r) { return r.json(); }).then(function (d) {
    data = d; render(input.value);
  });
  function score(item, terms) {
    var hay = (item.title + " " + item.summary + " " + (item.tags || []).join(" ")).toLowerCase();
    var s = 0;
    for (var i = 0; i < terms.length; i++) {
      var idx = hay.indexOf(terms[i]);
      if (idx === -1) return 0;
      s += item.title.toLowerCase().indexOf(terms[i]) !== -1 ? 3 : 1;
    }
    return s;
  }
  function render(q) {
    q = (q || "").trim().toLowerCase();
    results.innerHTML = "";
    if (!q) { empty.hidden = true; return; }
    var terms = q.split(/\\s+/);
    var matches = data.map(function (it) { return { it: it, s: score(it, terms) }; })
      .filter(function (m) { return m.s > 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 30);
    empty.hidden = matches.length !== 0;
    matches.forEach(function (m) {
      var li = document.createElement("li");
      li.className = "post-card";
      var tags = (m.it.tags || []).map(function (t) { return "#" + t; }).join(" ");
      li.innerHTML = '<div class="post-card__body"><h3><span class="muted">' +
        m.it.date + '</span> · <a href="' + m.it.url + '">' + m.it.title +
        '</a></h3><p>' + (m.it.summary || "") + '</p><div class="tag-row muted">' +
        tags + '</div></div>';
      results.appendChild(li);
    });
  }
  input.addEventListener("input", function () { render(input.value); });
})();
</script>`),
  });
};

function toc_from_content(content: HtmlString): HtmlString {
  const re =
    /<h([23])\b[^>]*>\s*<a href="#([^"]+)">([\s\S]*?)<\/a>\s*<\/h\1>/g;
  const entries: { level: number; id: string; text: string }[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(content.value)) !== null) {
    const text = m[3].replace(/<[^>]+>/g, "").trim();
    if (text) entries.push({ level: Number(m[1]), id: m[2], text });
  }
  if (entries.length < 2) return new HtmlString("");
  const items = entries.map((e) =>
    html`<li class="toc-l${e.level}"><a href="#${e.id}">${e.text}</a></li>`
  );
  return html`
    <details class="toc" open>
      <summary>Contents</summary>
      <ul>${items}</ul>
    </details>
  `;
}

export function post(post: Post, spellcheck: boolean): HtmlString {
  const meta = html`${time(post.date)} · ${post.readingTime} min read`;
  return base({
    src: post.src,
    title: post.title,
    description: post.summary,
    path: post.path,
    image: post.image,
    content: html`
      <div class="theme-banner">
        <span>Filed under</span>
        <a class="theme-pill" href="${post.theme.path}">
          ${post.theme.title}
        </a>
        <span class="post-meta">${meta}</span>
      </div>
      ${toc_from_content(post.content)}
      <article ${
      spellcheck ? 'contentEditable="true"' : ""
    }>\n${post.content}</article>
      ${tag_pills(post.tags)}
    `,
  });
}

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
          ${tag_pills(post.tags)}
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

function news_date(iso: string): HtmlString {
  const date = new Date(`${iso}T00:00:00Z`);
  return time(date);
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
<subtitle>${blurb}</subtitle>
<author><name>${author}</name></author>
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
<author><name>${author}</name></author>
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
