# Recovering the app.playbookiq.ai + playbookiq.ai source

The front end is **not in this repository**. `BUSINESS_MODEL.md` flags it as the company's
core unlocated asset. This is the runbook for getting it back.

## Confirmed facts (18 Sep)

The live UI is a **custom-designed front end, not Streamlit** (confirmed by Muwafag on sight).
That is decisive: Firebase Hosting serves static files, so the deployed site is a static
build — HTML/CSS/JS — and **a deployed static site is its own source**. No container
extraction needed.

The `streamlit_app.py` in this repo ("Soccer Analytics Demo", no phase tabs, no Retrait
notifications, no KEMO) is a separate, older prototype — not what is live. The earlier
assumption that the live product was Streamlit appears to be wrong.

Searched and ruled out:
- All 5 repos on the GitHub account; no HTML/JS/JSX/`firebase.json` in any of the 92 commits
  on any branch (`claude/setup-gpu-video-testing-JhgUH`, `final-deliverable`).
- GitHub code search for "playbookiq" across the account — 0 hits.
- All 13 Claude sessions — every football session points at THIS repo as its only source.
- All 4 published Claude artifacts — none is the dashboard or landing page.
- claude.ai chat history — searched by Muwafag, conversations not found.

What IS safe in this repo: `main.py` (CV pipeline), `vision/`, `geometry/`, `io_utils/`,
`api.py` (FastAPI backend), `auth.py` (Firebase token verification). **Only the front end is
missing.**

---

## Route 1 — download the deployed site (do this first)

```bash
wget -r -k -p -np -e robots=off --adjust-extension https://playbookiq.ai
wget -r -k -p -np -e robots=off --adjust-extension https://app.playbookiq.ai
```

This yields the complete production build. Then recover readable source from source maps:

```bash
# Look for maps shipped alongside the bundles
find . -name "*.map"
# Or check what the bundle points at
grep -r "sourceMappingURL" --include="*.js" . | tail
```

If `app.js.map` / `index-*.js.map` exist, reconstruct the original tree:

```bash
npx source-map-explorer dist/assets/*.js        # inspect
npx unmap  <bundle>.js.map  -o ./recovered-src  # extract original files
```

A shipped source map contains the **original, unminified, commented source of every component**
— effectively the repository the site was built from.

If maps were NOT shipped, the minified bundle is still fully functional code; it can be
beautified (`npx prettier --write`) and it preserves all logic, copy, styling and the KEMO
prompt strings. Recovering behaviour is guaranteed; recovering original variable names is not.

Also check Hosting release history: Firebase console → Hosting → the site → past releases.
Every previous deploy is retained and can be rolled back or inspected.

## Route 2 — the local machine

```bash
find ~ -name "firebase.json" -not -path "*/node_modules/*" 2>/dev/null
find ~ -name ".firebaserc" 2>/dev/null
grep -rl "playbookiq" ~ --include="*.json" --include="*.js" --include="*.tsx" 2>/dev/null

# Claude Code keeps a full JSONL transcript of every session, per project directory:
ls ~/.claude/projects/
grep -rl "playbookiq\|KEMO" ~/.claude/projects/ 2>/dev/null
```

`~/.claude/projects/` is the highest-value target: the directory names encode where the
project lived on disk even if the folder was deleted, and the transcripts contain the code
Claude wrote, verbatim.

## Route 3 — if the front end calls a backend

The live dashboard must fetch data from somewhere. Open the deployed bundle and search for
the API origin:

```bash
grep -roh "https://[a-z0-9.-]*\.run\.app[^\"']*" . | sort -u
grep -roh "https://[a-z0-9.-]*cloudfunctions[^\"']*" . | sort -u
```

Any Cloud Run / Cloud Functions URL found there points at the deployed backend, whose source
is recoverable from its container image or from the Cloud Build source tarball in
`gs://<PROJECT_ID>_cloudbuild/source/`.

---

## When recovered

```bash
git checkout -b frontend
mkdir -p frontend && cp -r <recovered>/* frontend/
git add frontend && git commit -m "Add app.playbookiq.ai front end to version control"
git push -u origin frontend
```

Until then the product exists only as a running deployment, and a single accidental
`firebase deploy` of an empty directory would destroy it.
