"""
Stage 1: download_data
Downloads Burgers equation dataset from URL specified in config.
DVC tracks the output .mat file by hash — guarantees reproducibility.
"""
import urllib.request
from pathlib import Path

import yaml


def main():
    with open("configs/burgers_fno.yaml") as f:
        cfg = yaml.safe_load(f)

    url = cfg["data"]["url"]
    path = Path(cfg["data"]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"Already exists: {path}")
        return

    print(f"Downloading from {url} ...")

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(url, path)
            print(f"Saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if path.exists():
                path.unlink()
            if attempt == max_retries:
                raise RuntimeError(
                    f"Download failed after {max_retries} attempts.\n"
                    f"Download manually from:\n  {url}\n"
                    f"Place file at: {path}"
                ) from e
            print("Retrying in 5 seconds...")
            import time

            time.sleep(5)


if __name__ == "__main__":
    main()
