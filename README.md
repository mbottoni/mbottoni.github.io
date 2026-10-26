# mbottoni.github.io

Source code for the blog. The `./src` directory contains a deno script that reads `.djot` from
`./content` and writes `.html` to `./out`.


$ deno task build
$ deno task watch
$ deno task check   # verify internal links, #anchors and assets in out/res
```

## Authoring a New Post

1. Create a `.dj` file inside `content/posts/` following the naming scheme  
   `YYYY-MM-DD-slug.dj` (e.g. `2025-08-04-moe.dj`). The build script infers the date and URL from the filename.
2. Write your post in Djot/Markdown syntax. Any embedded assets should live under `content/assets/`, inside the subfolder that matches the chosen theme (`frontier/`, `generative/`, `graph_rl/`, or `foundations/`) so they remain organized and get copied to `out/res/assets/` with the same structure.
3. Assign the post to one of the homepage subtopics by listing its slug in `src/themes.ts`. Each theme owns an array of post slugs; add your slug to the appropriate array so the post appears under that theme. If you skip this step the post defaults to the “Foundations” theme.
4. Run `deno task build` (or `deno task watch` while editing) to regenerate HTML and theme pages under `out/res/`.
5. Preview locally by serving `out/res/` (e.g. `deno task serve`) before committing and pushing.

## Series

Multi-part posts can be grouped in `src/series.ts`. Each entry lists its posts in
reading order; every listed post renders a "Part N of M" box linking to the others.
Refer to posts by slug, or by the full `YYYY-MM-DD-slug` stem when two posts share a
slug. The build fails if a series references an unknown or ambiguous post.

## Citations

Cite papers in a post with `[@key]`, or `[@a; @b]` for several at once (keep a
citation on one line):

```
Frankle and Carbin [@frankle2019] showed that ...
```

Entries live in `src/refs.ts`, each with a key, authors, title, year, venue and an
optional arXiv ID, DOI or URL. Citations render as numbered superscripts `[1]` in
order of first use, with the full reference as a hover tooltip, and the post ends
with a "References" list whose entries link back to the first citation. The build
fails if a post cites a key that is not in `src/refs.ts`. Copy author names and IDs
from the paper's arXiv page rather than from memory.

## Concept Map and Prerequisites

`src/prereqs.ts` lists "read A before B" edges between posts, using the same refs as
series (slug, or full `YYYY-MM-DD-slug` stem when ambiguous). Each post page renders a
"Before reading this" line under the series box and a "Read next" line above the related
posts. The build fails on an unknown or ambiguous ref, a self-edge, or a cycle.

The same edges feed `/map.html`, an interactive force-directed graph of every post:
nodes are coloured by theme and sized by reading time; arrows are prerequisites, dashed
lines join consecutive series parts, and faint lines (off by default) join posts that
share a tag. Hover highlights neighbours, click opens a post (tap once to select on
touch screens), drag moves a node, and scroll / pinch / the corner buttons zoom. The
build writes the graph to `/map.json`; `content/js/map.js` draws it with no external
libraries and is loaded only on that page. The page also renders the prerequisites as a
plain list, which is what screen readers and no-JS visitors get.

## Comments

Post pages embed [giscus](https://giscus.app) threads backed by GitHub Discussions
(category *General*). Configuration lives in `src/comments.ts`; add a slug to
`exclude` to hide comments on a specific post, or set `enabled: false` to turn them
off site-wide. The giscus GitHub App must be installed on the repository.

## Interactive Widgets

Posts can embed interactive figures. Drop a self-contained script at
`content/widgets/<name>.js` and reference it from a post with a fenced div:

```
{cap="Optional caption rendered under the figure."}
::: widget-<name>
:::
```

The build renders a `<figure class="widget" data-widget="<name>">` containing an
empty `.widget-mount`, and adds `<script defer src="/widgets/<name>.js">` **only
to pages that embed it** — no other post pays for the script. The widget script
is responsible for finding its own mount(s) and building the UI inside them:

```js
document.querySelectorAll('.widget[data-widget="<name>"] .widget-mount')
```

If a post references a widget with no matching file, the build fails rather than
shipping a dead mount point.

Shared chrome (`.widget-btn`, `.widget-stat`, `.widget-bar`, `.widget-canvas`,
`.widget-controls`, …) lives in `content/css/main.css`, so widgets look
consistent without shipping their own styles. Read theme colours from the CSS
custom properties (`--accent`, `--card-bg`, `--border`) at draw time and observe
`data-theme` on `<html>` so the figure follows the dark-mode toggle.

`content/widgets/hopfield.js` is the reference implementation.

## Runnable Code Blocks

Add `{.run}` on the line before a ` ```python ` or ` ```javascript ` fence to make it
executable in the reader's browser:

```
{.run}
```python
import numpy as np
print(np.arange(3))
```
```

The block renders with **Run** / **Edit** buttons and an output panel. Python runs
on [Pyodide](https://pyodide.org) (loaded from the jsDelivr CDN on first run, ~10 MB;
packages such as `numpy`/`scipy` are resolved from the snippet's imports), JavaScript
runs natively; both execute in a Web Worker so the page never blocks, and *Stop*
terminates a runaway snippet. Python state persists between runs on a page, like a
notebook kernel. `content/js/runner.js` is loaded only on pages that contain a
runnable block. Other languages fail the build.

## Updating the Resume

The resume PDF is generated from LaTeX source in `resume/`:

- `resume/resume.tex` — the resume content (edit this).
- `resume/resume.cls` — the document class / styling.
- `resume/build.sh` — compiles the source and overwrites `content/resume.pdf`.

To update the published resume:

1. Edit `resume/resume.tex`.
2. Run `deno task resume` (or `./resume/build.sh`) to recompile `content/resume.pdf`.
   Requires a LaTeX toolchain providing `pdflatex` (e.g. TeX Live / MacTeX).
3. Run `deno task build` so the refreshed `content/resume.pdf` is copied to `out/res/resume.pdf`, then commit and push.

The HTML version of the resume lives separately in `content/resume.dj`; keep it in sync when you change the PDF.