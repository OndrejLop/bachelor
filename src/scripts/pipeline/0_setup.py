#!/usr/bin/env python3
"""
Step 0 — minimal environment setup from a fresh git clone.

Two downloads, both idempotent (skips if target already present):
  1. P2Rank 2.5.1 tarball   -> data/tools/P2Rank/
  2. CESNET ownCloud bundle -> data/{filename}

Anything else (PDB structures, prankweb bulk data, model placement) is
out of scope here and handled by the user or by other scripts.
"""
import argparse
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOLS_DIR = ROOT / 'data' / 'tools'
P2RANK_DIR = TOOLS_DIR / 'P2Rank'

P2RANK_URL = "https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz"
CESNET_URL = "https://owncloud.cesnet.cz/index.php/s/sAvrV3RiTWKFmc1/download"


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def stage_p2rank():
    prank = P2RANK_DIR / "prank"
    if prank.exists():
        print(f"[p2rank] already installed at {P2RANK_DIR}, skipping", flush=True)
        return
    P2RANK_DIR.mkdir(parents=True, exist_ok=True)
    tarball = TOOLS_DIR / Path(P2RANK_URL).name
    if not tarball.exists():
        print(f"[p2rank] downloading {P2RANK_URL}", flush=True)
        run(["wget", "-c", "--tries=20", "--timeout=60", "--waitretry=10",
             "-O", str(tarball), P2RANK_URL])
    print(f"[p2rank] extracting into {P2RANK_DIR}", flush=True)
    # Tarball contains a top-level p2rank_2.5.1/ dir; --strip-components=1
    # flattens that so files land directly under P2RANK_DIR.
    run(["tar", "-xzf", str(tarball), "-C", str(P2RANK_DIR), "--strip-components=1"])
    if prank.exists():
        prank.chmod(prank.stat().st_mode | 0o111)
        print(f"[p2rank] installed: {prank}", flush=True)
    else:
        print(f"[p2rank] WARNING: {prank} not found after extraction", flush=True)


def _filename_from_url(url):
    """Try to get a sensible filename from the response, fall back to URL stem."""
    try:
        with urlopen(url, timeout=60) as r:
            cd = r.headers.get("Content-Disposition", "")
            if "filename=" in cd:
                name = cd.split("filename=", 1)[1].strip().strip('"').split(";")[0].strip()
                if name:
                    return name
    except Exception:
        pass
    return "cesnet_bundle"


def stage_cesnet():
    data_dir = ROOT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    name = _filename_from_url(CESNET_URL)
    target = data_dir / name
    if target.exists() and target.stat().st_size > 0:
        print(f"[cesnet] already present at {target}, skipping", flush=True)
        return
    print(f"[cesnet] downloading {CESNET_URL} -> {target}", flush=True)
    run(["wget", "-c", "--tries=20", "--timeout=60", "--waitretry=10",
         "-O", str(target), CESNET_URL])
    print(f"[cesnet] saved {target.stat().st_size / 1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-p2rank", action="store_true")
    ap.add_argument("--skip-cesnet", action="store_true")
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print(f"Pipeline setup (step 0)\nProject root: {ROOT}", flush=True)
    print("=" * 60, flush=True)

    if not args.skip_p2rank:
        stage_p2rank()
    if not args.skip_cesnet:
        stage_cesnet()

    print("\nSetup complete.", flush=True)


if __name__ == "__main__":
    main()
