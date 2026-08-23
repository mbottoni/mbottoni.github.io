// Post-build link and asset checker. Walks every HTML file in out/res and
// verifies that each internal href/src (and same-page #fragment) resolves to
// a generated file / element id. Exits non-zero on the first broken batch so
// CI fails before deploying.
//
//   deno task check

import * as fs from "std/fs/mod.ts";
import * as path from "std/path/mod.ts";

const root = "out/res";

type Problem = { page: string; target: string; reason: string };

// External hosts are not fetched: keep the check hermetic and fast.
function is_external(url: string): boolean {
  return /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(url);
}

const ATTR_RE = /\b(?:href|src|poster)\s*=\s*"([^"]*)"/gi;
const ID_RE = /\bid\s*=\s*"([^"]*)"/gi;

async function file_exists(p: string): Promise<boolean> {
  try {
    return (await Deno.stat(p)).isFile;
  } catch {
    return false;
  }
}

/** Map a site path to an output file, treating `/dir/` as `/dir/index.html`. */
async function resolve_target(site_path: string): Promise<string | undefined> {
  const candidates = [site_path];
  if (site_path.endsWith("/")) candidates.push(site_path + "index.html");
  for (const c of candidates) {
    const p = path.join(root, c);
    if (await file_exists(p)) return p;
  }
  return undefined;
}

async function main() {
  const pages: string[] = [];
  for await (const e of fs.walk(root, { exts: [".html"], includeDirs: false })) {
    pages.push(e.path);
  }

  // ids per page, filled lazily so fragment checks across pages work.
  const ids = new Map<string, Set<string>>();
  const page_ids = async (p: string): Promise<Set<string>> => {
    let set = ids.get(p);
    if (set) return set;
    set = new Set<string>();
    const html = await Deno.readTextFile(p);
    for (const m of html.matchAll(ID_RE)) set.add(m[1]);
    ids.set(p, set);
    return set;
  };

  const problems: Problem[] = [];
  let checked = 0;

  for (const page of pages) {
    // Inline scripts build URLs from JS strings; skip them.
    const html = (await Deno.readTextFile(page))
      .replace(/<script\b[\s\S]*?<\/script>/gi, "");
    const site_page = page.slice(root.length).split(path.SEP).join("/");
    for (const m of html.matchAll(ATTR_RE)) {
      const raw = m[1].trim();
      if (raw === "" || raw.startsWith("#") && raw.length === 1) continue;
      if (is_external(raw) || raw.startsWith("data:") || raw.startsWith("mailto:")) continue;
      checked++;

      const [without_query] = raw.split("?");
      const [target_path, fragment] = without_query.split("#");

      let file: string | undefined;
      if (target_path === "") {
        file = page; // same-page fragment
      } else {
        const abs = target_path.startsWith("/")
          ? target_path
          : path.posix.join(path.posix.dirname(site_page), target_path);
        file = await resolve_target(decodeURIComponent(abs));
        if (!file) {
          problems.push({ page: site_page, target: raw, reason: "missing file" });
          continue;
        }
      }

      if (fragment && file.endsWith(".html")) {
        const set = await page_ids(file);
        if (!set.has(decodeURIComponent(fragment))) {
          problems.push({ page: site_page, target: raw, reason: "missing #id" });
        }
      }
    }
  }

  // The source links in the page footer point at the repo, which is fine to
  // skip, but content/ assets referenced from posts must have been copied.
  if (problems.length) {
    console.error(`${problems.length} broken internal reference(s):`);
    for (const p of problems) {
      console.error(`  ${p.page}  ->  ${p.target}  (${p.reason})`);
    }
    Deno.exit(1);
  }
  console.log(`checked ${checked} internal references across ${pages.length} pages: ok`);
}

if (import.meta.main) await main();
