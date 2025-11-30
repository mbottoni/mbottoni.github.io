# mbottoni.github.io

Source code for the blog. The `./src` directory contains a deno script that reads `.djot` from
`./content` and writes `.html` to `./out`.


$ deno task build
$ deno tasks serve
$ deno task watch
```

## Authoring a New Post

1. Create a `.dj` file inside `content/posts/` following the naming scheme  
   `YYYY-MM-DD-slug.dj` (e.g. `2025-08-04-moe.dj`). The build script infers the date and URL from the filename.
2. Write your post in Djot/Markdown syntax. Any embedded assets should live under `content/assets/`, inside the subfolder that matches the chosen theme (`frontier/`, `generative/`, `graph_rl/`, or `foundations/`) so they remain organized and get copied to `out/res/assets/` with the same structure.
3. Assign the post to one of the homepage subtopics by listing its slug in `src/themes.ts`. Each theme owns an array of post slugs; add your slug to the appropriate array so the post appears under that theme. If you skip this step the post defaults to the “Foundations” theme.
4. Run `deno task build` (or `deno task watch` while editing) to regenerate HTML and theme pages under `out/res/`.
5. Preview locally by serving `out/res/` (e.g. `deno task serve`) before committing and pushing.

## Math / LaTeX

- Inline math: wrap expressions with `\( ... \)` or `$...$`.  
- Display math: use `\[ ... \]` or `$$...$$`.  
- KaTeX assets are vendored under `content/assets/vendor/katex/` and are loaded on every page, so math renders client-side without extra build steps.
