# Recovering the app.playbookiq.ai + playbookiq.ai source

Status: the front end is **not in this repository**. `BUSINESS_MODEL.md` flags this as the
company's core unlocated asset. This file is the runbook for getting it back.

## What was already confirmed (18 Sep)

Searched and ruled out:
- All 5 repos on the GitHub account — only this one is football; no HTML/JS/JSX/`firebase.json`
  in any of the 92 commits, on any branch (`claude/setup-gpu-video-testing-JhgUH`, `final-deliverable`).
- GitHub code search for "playbookiq" across the account — 0 hits.
- All 13 Claude sessions on the account — every football session points at THIS repo as its
  only source. No session has the front end attached.
- All 4 published Claude artifacts — "First Touch" (learning tracker), 2x Rocío
  architecture, 1 water dashboard. None is the dashboard or landing page.

What IS here: `main.py` (CV pipeline), `vision/`, `geometry/`, `io_utils/`, `api.py` (FastAPI),
`auth.py` (Firebase token verification), and `streamlit_app.py` — but that Streamlit file is
titled "Soccer Analytics Demo" and contains **no phase tabs, no Retrait notifications, no KEMO**.
It is an earlier, simpler app than what is live.

## Route 1 — Cloud Run container (highest probability)

Firebase Hosting serves static files only; it cannot run Python. If the live app is really
Streamlit, Hosting must be rewriting to **Cloud Run**, and the container image in Artifact
Registry still holds the complete source.

```bash
gcloud auth login
gcloud projects list                      # find the playbookiq project id
gcloud config set project <PROJECT_ID>

gcloud run services list --platform=managed
gcloud run services describe <SERVICE> --region=<REGION> \
  --format='value(spec.template.spec.containers[0].image)'
```

Then extract the source from the image:

```bash
gcloud auth configure-docker <REGION>-docker.pkg.dev
docker pull <IMAGE>
docker create --name recover <IMAGE>
docker cp recover:/app ./recovered-app      # /app, /src or /home/app — check the Dockerfile layer
docker rm recover
```

No Docker available? `gcloud run services describe` also shows the build that produced the
image — check **Cloud Build history** in the console, which stores the uploaded source tarball
in a `gs://<project>_cloudbuild/source/` bucket:

```bash
gsutil ls gs://<PROJECT_ID>_cloudbuild/source/
gsutil cp gs://<PROJECT_ID>_cloudbuild/source/<newest>.tgz .
tar xzf <newest>.tgz
```

That tarball is the exact source directory as it existed on the machine that deployed it.

## Route 2 — Firebase Hosting static files (for the landing page)

If `playbookiq.ai` is a static build (React/HTML from a Claude design), the deployed files are
fetchable directly:

```bash
wget -r -k -p -np -e robots=off https://playbookiq.ai
```

Then look for **source maps** (`*.js.map`) next to the bundles. Vite and CRA builds ship them
unless explicitly disabled — a `.map` file reconstructs the original, unminified component
source. Check `dist/assets/*.js.map`, or view-source for a `//# sourceMappingURL=` comment.

Hosting also keeps every past release: console → Hosting → the site → release history.

## Route 3 — the local machine

```bash
# macOS / Linux
find ~ -name "firebase.json" -not -path "*/node_modules/*" 2>/dev/null
find ~ -name ".firebaserc" 2>/dev/null
grep -rl "playbookiq" ~ --include="*.json" --include="*.py" --include="*.js" 2>/dev/null

# Claude Code keeps every session transcript locally, including files it wrote:
grep -rl "playbookiq\|KEMO" ~/.claude/projects/ 2>/dev/null
ls ~/.claude/projects/
```

`~/.claude/projects/` is the important one — Claude Code stores full JSONL transcripts per
project directory, and the directory names reveal where the project lived on disk even if the
folder itself was deleted.

## Route 4 — the Claude conversations

The build sessions were on claude.ai, not in this repo's sessions. Search claude.ai chat
history for "playbookiq", "KEMO", "Retrait". Any artifact created inside a chat is recoverable
from that conversation, and Claude Code sessions on the desktop app are listed under the
session picker.

## When recovered — do this immediately

```bash
git checkout -b frontend
mkdir frontend && cp -r <recovered> frontend/
git add frontend && git commit -m "Add app.playbookiq.ai front end to version control"
git push -u origin frontend
```

Until then the product exists only as a running deployment.
