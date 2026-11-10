"""Download the public DFL tracking dataset (Bassek et al., Scientific Data 2025; CC-BY 4.0).

figshare article 28196177: 7 Bundesliga matches, 21 files, about 2.6 GB, into external/dfl/.
Files already present with the right size are skipped; every file is MD5-checked.
"""
import hashlib
import json
import sys
import urllib.request

from possession.paths import EXTERNAL

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
UA = {"User-Agent": "possession-pipeline/0.1"}


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out = EXTERNAL / "dfl"
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(ARTICLE, headers=UA)) as r:
        files = json.load(r)["files"]
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB")
    bad = 0
    for f in files:
        dest = out / f["name"]
        if not (dest.exists() and dest.stat().st_size == f["size"]):
            tmp = dest.with_suffix(dest.suffix + ".part")
            req = urllib.request.Request(f["download_url"], headers=UA)
            with urllib.request.urlopen(req) as r, open(tmp, "wb") as w:
                while chunk := r.read(1 << 20):
                    w.write(chunk)
            tmp.replace(dest)
        ok = md5(dest) == f["computed_md5"]
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {f['name']} {f['size'] / 1e6:.0f} MB", flush=True)
    print(f"done: {len(files) - bad}/{len(files)} MD5 ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
