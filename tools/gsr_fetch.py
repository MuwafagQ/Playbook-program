"""Fetch SoccerNet GSR ground truth and the baseline's published results, without images.

  python -m tools.gsr_fetch labels --split valid --out data/SoccerNetGS          # ~290 MB download
  python -m tools.gsr_fetch baseline --split valid --out data/SoccerNetGS        # ~1 GB tracker state
  python -m tools.gsr_fetch frames --split valid --seqs SNGS-021 --out data/SoccerNetGS  # ~160 MB/clip -> mp4

The split zips are mostly images (valid.zip is 11 GB). Scoring only needs each clip's
Labels-GameState.json, so `labels` reads just those entries out of the remote zip with HTTP
range requests. Uses SoccerNet's public share credentials (the same ones the SoccerNet pip
package uses); the data is for research and benchmarking, not for training a shipped model.
Needs network access to exrcsdrive.kaust.edu.sa (labels) and zenodo.org (baseline).
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile

import requests

WEBDAV = "https://exrcsdrive.kaust.edu.sa/public.php/webdav/{split}.zip"
AUTH = ("iOJmJH6rYnx7mOS", "SoccerNet")
BASELINE = {
    "valid": "https://zenodo.org/records/11065177/files/gamestate-prtreid-strongsort-valid-compressed.pklz?download=1",
    "test": "https://zenodo.org/records/11065177/files/gamestate-prtreid-strongsort-test-compressed.pklz?download=1",
}


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, fetched in blocks with HTTP Range."""

    def __init__(self, url, auth=None, block=1 << 20, max_blocks=64):
        self.url, self.auth, self.block, self.max_blocks = url, auth, block, max_blocks
        self.s = requests.Session()
        self.size = int(self.s.head(url, auth=auth, timeout=60).headers["Content-Length"])
        self.pos, self.cache, self.fetched = 0, {}, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            a = i * self.block
            b = min(self.size, a + self.block) - 1
            r = self.s.get(self.url, auth=self.auth, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            if r.status_code != 206:
                raise RuntimeError(f"server did not honour a range request (HTTP {r.status_code})")
            self.cache[i] = r.content
            self.fetched += len(r.content)
            if len(self.cache) > self.max_blocks:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        out = bytearray()
        while len(out) < n:
            i, o = divmod(self.pos + len(out), self.block)
            out += self._blk(i)[o:o + n - len(out)]
        buf[:n] = out
        self.pos += n
        return n


def fetch_labels(split: str, out: str, seqs: list[str] | None = None) -> list[str]:
    f = HttpRangeFile(WEBDAV.format(split=split), AUTH)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = [n for n in z.namelist() if n.endswith("Labels-GameState.json")]
    if seqs:
        names = [n for n in names if n.split("/")[0] in set(seqs)]
    done = []
    for n in sorted(names):
        dst = os.path.join(out, split, n)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            with z.open(n) as src, open(dst + ".part", "wb") as w:
                w.write(src.read())
            os.replace(dst + ".part", dst)
        done.append(n.split("/")[0])
    print(f"{len(done)} label files in {out}/{split}; downloaded {f.fetched / 1e6:.0f} MB of {f.size / 1e9:.1f} GB")
    return done


def fetch_frames(split: str, out: str, seqs: list[str], fps: int = 25, keep_jpg: bool = False) -> list[str]:
    """Fetch clips' frames (img1/*.jpg) and write <out>/<split>/<seq>.mp4 (frame i = image i+1)."""
    import shutil
    import subprocess

    try:
        ffmpeg = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg is needed to build the clip videos") from e
    f = HttpRangeFile(WEBDAV.format(split=split), AUTH)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = z.namelist()
    videos = []
    for seq in seqs:
        video = os.path.join(out, split, f"{seq}.mp4")
        if os.path.exists(video):
            videos.append(video)
            continue
        # read in storage order: the zip is not sorted by name, and following name order
        # makes the range reader fetch the same blocks many times over
        imgs = sorted((n for n in names if n.startswith(f"{seq}/img1/") and n.endswith(".jpg")),
                      key=lambda n: z.getinfo(n).header_offset)
        if not imgs:
            raise FileNotFoundError(f"no frames for {seq} in the {split} zip")
        img_dir = os.path.join(out, split, seq, "img1")
        os.makedirs(img_dir, exist_ok=True)
        for n in imgs:
            dst = os.path.join(img_dir, os.path.basename(n))
            if not os.path.exists(dst):
                with z.open(n) as src, open(dst, "wb") as w:
                    w.write(src.read())
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-start_number", "1", "-i", os.path.join(img_dir, "%06d.jpg"),
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
                        video + ".part.mp4"], check=True)
        os.replace(video + ".part.mp4", video)
        if not keep_jpg:
            shutil.rmtree(img_dir)
        print(f"{seq}: {len(imgs)} frames -> {video}")
        videos.append(video)
    print(f"downloaded {f.fetched / 1e6:.0f} MB")
    return videos


def fetch_baseline(split: str, out: str) -> str:
    dst = os.path.join(out, f"baseline-{split}.pklz")
    if not os.path.exists(dst):
        os.makedirs(out, exist_ok=True)
        with requests.get(BASELINE[split], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dst + ".part", "wb") as w:
                for chunk in r.iter_content(1 << 22):
                    w.write(chunk)
        os.replace(dst + ".part", dst)
    print(f"baseline tracker state: {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)")
    return dst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["labels", "baseline", "frames"])
    p.add_argument("--split", default="valid")
    p.add_argument("--out", default="data/SoccerNetGS")
    p.add_argument("--seqs", nargs="*", default=None, help="labels/frames: just these clips (required for frames)")
    p.add_argument("--keep-jpg", action="store_true", help="frames: keep the images next to the video")
    a = p.parse_args(argv)
    if a.what == "labels":
        fetch_labels(a.split, a.out, a.seqs)
    elif a.what == "frames":
        if not a.seqs:
            p.error("frames needs --seqs")
        fetch_frames(a.split, a.out, a.seqs, keep_jpg=a.keep_jpg)
    else:
        fetch_baseline(a.split, a.out)


if __name__ == "__main__":
    main()
