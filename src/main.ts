import * as async from "std/async/mod.ts";
import * as fs from "std/fs/mod.ts";
import * as templates from "./templates.ts";
import * as djot from "./djot.ts";
import { HtmlString } from "./templates.ts";
import { resolveTheme, themes, type ThemeConfig } from "./themes.ts";
import { projects } from "./projects.ts";
import { news } from "./news.ts";
import { resolveTags, tagSlug } from "./tags.ts";
import { resolveSeries, type SeriesMembership, validateSeries } from "./series.ts";

async function main() {
  const params = {
    update: false,
    spell: false,
    profile: false,
    filter: "",
  };

  const subcommand = Deno.args[0];

  let i = 1;
  for (; i < Deno.args.length; i++) {
    switch (Deno.args[i]) {
      case "--update": {
        params.update = true;
        break;
      }
      case "--spell": {
        params.spell = true;
        break;
      }
      case "--profile": {
        params.profile = true;
        break;
      }
      case "--filter": {
        params.filter = Deno.args[i + 1] ?? "";
        i++;
        break;
      }
      default:
        fatal(`unexpected argument: ${Deno.args[i]}`);
    }
  }

  if (subcommand === "build") {
    await build(params);
  } else if (subcommand === "watch") {
    await watch(params);
  } else {
    fatal("subcommand required");
  }
}

function fatal(message: string) {
  console.error(message);
  Deno.exit(1);
}

async function watch(params: { filter: string }) {
  let signal = async.deferred();
  (async () => {
    let build_id = 0;
    while (await signal) {
      signal = async.deferred();
      console.log(`rebuild #${build_id}`);
      build_id += 1;
      await build({
        update: true,
        spell: false,
        profile: false,
        filter: params.filter,
      });
    }
  })();

  signal.resolve(true);

  const rebuild_debounced = async.debounce(
    () => signal.resolve(true),
    16,
  );

  for await (const event of Deno.watchFs("./content", { recursive: true })) {
    if (event.kind == "access") continue;
    await rebuild_debounced();
  }
  signal.resolve(false);
}

class Ctx {
  constructor(
    public read_ms: number = 0,
    public parse_ms: number = 0,
    public render_ms: number = 0,
    public collect_ms: number = 0,
    public total_ms: number = 0,
  ) {}
}

async function build(params: {
  update: boolean;
  spell: boolean;
  profile: boolean;
  filter: string;
}) {
  const t = performance.now();

  const ctx = new Ctx();
  if (params.update) {
    await Deno.mkdir("./out/res", { recursive: true });
  } else {
    await fs.emptyDir("./out/res");
  }

  const posts = await collect_posts(ctx, params.filter);
  const themeGroups = themes.map((theme) => ({
    theme,
    posts: posts.filter((post) => post.theme.key === theme.key),
  })).filter((group) => group.posts.length > 0);

  await update_file(
    "out/res/index.html",
    templates.post_list(themeGroups, posts, news).value,
  );
  for (const group of themeGroups) {
    await update_file(
      `out/res${group.theme.path}`,
      templates.theme_page(group.theme, group.posts).value,
    );
  }
  await update_file("out/res/feed.xml", templates.feed(posts).value);

  // Projects: render a detail page for each project that has a
  // content/projects/<slug>.dj file; the index links to those when present.
  const projectDetails = new Set<string>();
  for (const project of projects) {
    let text: string | undefined;
    try {
      text = await Deno.readTextFile(`content/projects/${project.slug}.dj`);
    } catch {
      continue;
    }
    const parsed = djot.parse(text);
    const body = djot.render(parsed.doc, {}, parsed.math);
    await update_file(
      `out/res/projects/${project.slug}.html`,
      templates.project_detail_page(project, body).value,
    );
    projectDetails.add(project.slug);
  }
  await update_file(
    "out/res/projects.html",
    templates.projects_page(projects, projectDetails).value,
  );
  await update_file("out/res/news.html", templates.news_page(news).value);
  await update_file("out/res/archive.html", templates.archive_page(posts).value);

  // Tags: an index plus one page per tag
  const tagMap = new Map<string, Post[]>();
  for (const post of posts) {
    for (const tag of post.tags) {
      const list = tagMap.get(tag) ?? [];
      list.push(post);
      tagMap.set(tag, list);
    }
  }
  const tagCounts = [...tagMap.entries()].map(([tag, ps]) => ({
    tag,
    count: ps.length,
  }));
  await update_file(
    "out/res/tags.html",
    templates.tags_index_page(tagCounts).value,
  );
  for (const [tag, ps] of tagMap) {
    await update_file(
      `out/res/tags/${tagSlug(tag)}.html`,
      templates.tag_page(tag, ps).value,
    );
  }

  // Arcade
  await update_file("out/res/arcade.html", templates.arcade_page().value);

  // Client-side search index + page
  await update_file("out/res/search.html", templates.search_page().value);
  const search_index = posts.map((post) => ({
    title: post.title,
    summary: post.summary,
    url: post.path,
    date: post.date.toISOString().slice(0, 10),
    tags: post.tags,
  }));
  await update_file("out/res/search.json", JSON.stringify(search_index));
  const nav = compute_nav(posts);
  for (const post of posts) {
    await update_file(
      `out/res${post.path}`,
      templates.post(post, params.spell, nav.get(post.path)!, posts).value,
    );
  }

  const pages = ["about", "resume", "links", "style", "repositories"];
  for (const page of pages) {
    const text = await Deno.readTextFile(`content/${page}.dj`);
    const parsed = djot.parse(text);
    const html = djot.render(parsed.doc, {}, parsed.math);
    await update_file(`out/res/${page}.html`, templates.page(page, html).value);
  }

  const paths = [
    "favicon.svg",
    "favicon.png",
    "resume.pdf",
    "css/*",
    "assets/*",
    "assets/resilient-parsing/*",
    "arcade/*",
    "widgets/*",
  ];
  for (const path of paths) {
    await update_path(path);
  }

  ctx.total_ms = performance.now() - t;
  console.log(`${ctx.total_ms}ms`);
  if (params.profile) console.log(JSON.stringify(ctx));
}

async function update_file(path: string, content: Uint8Array | string) {
  if (!content) return;
  await fs.ensureFile(path);
  await fs.ensureDir("./build");
  const temp = await Deno.makeTempFile({ dir: "./build" });
  if (content instanceof Uint8Array) {
    await Deno.writeFile(temp, content);
  } else {
    await Deno.writeTextFile(temp, content);
  }
  await Deno.rename(temp, path);
}

async function update_path(path: string) {
  if (path.endsWith("*")) {
    const dir = path.replace("*", "");
    const futs = [];
    for await (const entry of Deno.readDir(`content/${dir}`)) {
      if (entry.isFile) {
        futs.push(update_path(`${dir}/${entry.name}`));
      } else if (entry.isDirectory) {
        futs.push(update_path(`${dir}/${entry.name}/*`));
      }
    }
    await Promise.all(futs);
  } else {
    await update_file(
      `out/res/${path}`,
      await Deno.readFile(`content/${path}`),
    );
  }
}

export type Post = {
  year: number;
  month: number;
  day: number;
  slug: string;
  date: Date;
  title: string;
  path: string;
  src: string;
  content: HtmlString;
  summary: string;
  theme: ThemeConfig;
  tags: string[];
  readingTime: number;
  image?: string;
  widgets: string[];
  series?: SeriesMembership;
};

// Cross-post navigation computed once all posts are known: chronological
// neighbours, the neighbours within the same theme, and tag-similar posts.
export type PostNav = {
  newer?: Post;
  older?: Post;
  themeNewer?: Post;
  themeOlder?: Post;
  related: Post[];
};

export function compute_nav(posts: Post[]): Map<string, PostNav> {
  // `posts` is sorted newest first.
  const nav = new Map<string, PostNav>();
  for (let i = 0; i < posts.length; i++) {
    const post = posts[i];
    const sameTheme = posts.filter((p) => p.theme.key === post.theme.key);
    const j = sameTheme.indexOf(post);
    const tags = new Set(post.tags);
    const related = posts
      .filter((p) => p !== post)
      .map((p) => {
        const shared = p.tags.filter((t) => tags.has(t)).length;
        // Shared tags dominate; same theme is a tiebreaker so posts with no
        // tags still get sensible suggestions.
        const score = shared * 10 + (p.theme.key === post.theme.key ? 1 : 0);
        return { p, score };
      })
      .filter((it) => it.score > 0)
      .sort((a, b) =>
        b.score - a.score || b.p.date.getTime() - a.p.date.getTime()
      )
      .slice(0, 3)
      .map((it) => it.p);
    nav.set(post.path, {
      newer: posts[i - 1],
      older: posts[i + 1],
      themeNewer: sameTheme[j - 1],
      themeOlder: sameTheme[j + 1],
      related,
    });
  }
  return nav;
}

async function collect_posts(ctx: Ctx, filter: string): Promise<Post[]> {
  const start = performance.now();
  const posts = [];
  for await (
    const entry of fs.walk("./content/posts", { includeDirs: false })
  ) {
    if (!entry.name.endsWith(".dj")) continue;
    if (filter !== "") {
      if (entry.name.indexOf(filter) === -1) continue;
    }
    const [, y, m, d, slug] = entry.name.match(
      /^(\d\d\d\d)-(\d\d)-(\d\d)-(.*)\.dj$/,
    )!;
    const [year, month, day] = [y, m, d].map((it) => parseInt(it, 10));
    const date = new Date(Date.UTC(year, month - 1, day));

    let t = performance.now();
    const text = await Deno.readTextFile(entry.path);
    ctx.read_ms += performance.now() - t;

    t = performance.now();
    const parsed = djot.parse(text);
    ctx.parse_ms += performance.now() - t;

    t = performance.now();
    const render_ctx = {
      date,
      summary: undefined,
      title: undefined,
      widgets: new Set<string>(),
    };
    const html = djot.render(parsed.doc, render_ctx, parsed.math);
    ctx.render_ms += performance.now() - t;

    // Fail loudly rather than shipping a post with a dead widget mount.
    const widgets = [...render_ctx.widgets].sort();
    for (const widget of widgets) {
      if (!await fs.exists(`content/widgets/${widget}.js`)) {
        fatal(
          `${entry.name}: embeds '::: widget-${widget}' but ` +
            `content/widgets/${widget}.js does not exist`,
        );
      }
    }

    const hero = extract_first_image(text);
    const words = text.trim().split(/\s+/).filter(Boolean).length;
    const readingTime = Math.max(1, Math.round(words / 200));

    posts.push({
      year,
      month,
      day,
      slug,
      date,
      title: render_ctx.title!,
      content: html,
      summary: render_ctx.summary!,
      path: `/${y}/${m}/${d}/${slug}.html`,
      src: `/content/posts/${y}-${m}-${d}-${slug}.dj`,
      theme: resolveTheme(slug),
      tags: resolveTags(slug),
      readingTime,
      image: hero,
      widgets,
      series: resolveSeries(`${y}-${m}-${d}-${slug}`, slug),
    });
  }
  posts.sort((l, r) => l.path < r.path ? 1 : -1);
  if (filter === "") {
    validateSeries(
      posts.map((p) => `${p.year}-${pad(p.month)}-${pad(p.day)}-${p.slug}`),
      posts.map((p) => p.slug),
    );
  }
  ctx.collect_ms = performance.now() - start;
  return posts;
}

const pad = (n: number) => String(n).padStart(2, "0");

function extract_first_image(source: string): string | undefined {
  const markdown_image = source.match(/!\[[^\]]*\]\(([^\s)]+)(?:\s"[^"]*")?\)/);
  if (markdown_image) return markdown_image[1];
  return undefined;
}

if (import.meta.main) await main();
