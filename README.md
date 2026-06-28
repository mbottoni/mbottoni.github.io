# mbottoni.github.io

Source code for the blog. The `./src` directory contains a deno script that reads `.djot` from
`./content` and writes `.html` to `./out`.


$ deno task build
$ deno task watch
```

## Authoring a New Post

1. Create a `.dj` file inside `content/posts/` following the naming scheme  
   `YYYY-MM-DD-slug.dj` (e.g. `2025-08-04-moe.dj`). The build script infers the date and URL from the filename.
2. Write your post in Djot/Markdown syntax. Any embedded assets should live under `content/assets/`, inside the subfolder that matches the chosen theme (`frontier/`, `generative/`, `graph_rl/`, or `foundations/`) so they remain organized and get copied to `out/res/assets/` with the same structure.
3. Assign the post to one of the homepage subtopics by listing its slug in `src/themes.ts`. Each theme owns an array of post slugs; add your slug to the appropriate array so the post appears under that theme. If you skip this step the post defaults to the “Foundations” theme.
4. Run `deno task build` (or `deno task watch` while editing) to regenerate HTML and theme pages under `out/res/`.
5. Preview locally by serving `out/res/` (e.g. `deno task serve`) before committing and pushing.

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