#!/usr/bin/env bash
# Recover the playbookiq.ai / app.playbookiq.ai front end from the live deployment.
#
# The live UI is a custom static build on Firebase Hosting, so the deployed site
# IS the source. This mirrors both sites, extracts original source from any
# shipped source maps, and reports what it found.
#
# Requires outbound access to playbookiq.ai and app.playbookiq.ai.
# Usage: bash launchpad/tools/recover-frontend.sh [outdir]

set -uo pipefail

OUT="${1:-recovered-frontend}"
mkdir -p "$OUT"
cd "$OUT" || exit 1

echo "==> Checking reachability"
for host in playbookiq.ai app.playbookiq.ai; do
  code=$(curl -sS -o /dev/null -w "%{http_code}" -L --max-time 25 "https://$host" 2>/dev/null)
  echo "    https://$host -> HTTP ${code:-blocked}"
  if [ "${code:-000}" = "000" ]; then
    echo "    BLOCKED. Egress policy still denies this host; stop here and allowlist it."
    exit 2
  fi
done

echo "==> Mirroring both sites"
for host in playbookiq.ai app.playbookiq.ai; do
  wget -q --show-progress -r -k -p -np -e robots=off --adjust-extension \
       --no-host-directories --directory-prefix="site-$host" \
       "https://$host" 2>&1 | tail -2
done

echo "==> Inventory"
find . -type f \( -name '*.html' -o -name '*.js' -o -name '*.css' -o -name '*.map' \) \
  | sed 's/^/    /' | head -60
echo "    ---"
echo "    html: $(find . -name '*.html' | wc -l)  js: $(find . -name '*.js' | wc -l)  css: $(find . -name '*.css' | wc -l)  maps: $(find . -name '*.map' | wc -l)"

echo "==> Hunting for source maps"
# Maps referenced but not mirrored are still fetchable directly.
grep -roh "sourceMappingURL=[^ */]*" --include='*.js' --include='*.css' . 2>/dev/null \
  | sed 's/sourceMappingURL=//' | sort -u | while read -r m; do
    case "$m" in
      data:*) echo "    inline map found (embedded in bundle)";;
      *) echo "    referenced map: $m";;
    esac
done

for host in playbookiq.ai app.playbookiq.ai; do
  find "site-$host" -name '*.js' 2>/dev/null | while read -r js; do
    rel="${js#site-$host/}"
    curl -sfS --max-time 20 -o "${js}.map" "https://$host/${rel}.map" 2>/dev/null \
      && echo "    downloaded ${rel}.map"
  done
done

echo "==> Extracting original source from maps"
if find . -name '*.map' | grep -q .; then
  find . -name '*.map' | while read -r map; do
    node -e '
      const fs=require("fs"),path=require("path");
      const m=JSON.parse(fs.readFileSync(process.argv[1],"utf8"));
      if(!m.sources||!m.sourcesContent){console.log("    no embedded sources in "+process.argv[1]);process.exit(0)}
      const root=path.join("original-src");
      m.sources.forEach((s,i)=>{
        const c=m.sourcesContent[i]; if(c==null)return;
        const clean=s.replace(/^(webpack:\/\/|\.\.\/)+/,"").replace(/^\/+/,"").replace(/[?#].*$/,"");
        const dest=path.join(root,clean);
        fs.mkdirSync(path.dirname(dest),{recursive:true});
        fs.writeFileSync(dest,c);
      });
      console.log("    extracted "+m.sources.length+" files from "+path.basename(process.argv[1]));
    ' "$map"
  done
  echo "    -> original-src/ now holds the reconstructed source tree"
else
  echo "    No source maps. The bundles are still complete working code."
  echo "    Beautify them with: npx prettier --write \"**/*.js\""
fi

echo "==> Looking for the backend API the front end talks to"
grep -roh "https://[a-zA-Z0-9._-]*\.run\.app[^\"',) ]*"        . 2>/dev/null | sort -u | sed 's/^/    cloud-run: /'
grep -roh "https://[a-zA-Z0-9._-]*cloudfunctions[^\"',) ]*"     . 2>/dev/null | sort -u | sed 's/^/    functions: /'
grep -roh "https://[a-zA-Z0-9._-]*\.firebaseio\.com[^\"',) ]*"  . 2>/dev/null | sort -u | sed 's/^/    rtdb:      /'
grep -roh "[a-zA-Z0-9._-]*\.firebaseapp\.com"                   . 2>/dev/null | sort -u | sed 's/^/    authdomain:/'
grep -roh "\"projectId\"[: ]*\"[^\"]*\""                        . 2>/dev/null | sort -u | sed 's/^/    project:   /'

echo "==> Feature check (is this really the live dashboard?)"
for kw in KEMO Retrait retrait "Build-up" Transition Finishing xT xP; do
  n=$(grep -ril "$kw" . 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "    found '$kw' in $n file(s)"
done

echo
echo "Done. Review $OUT/original-src (if present), then commit to a frontend branch."
