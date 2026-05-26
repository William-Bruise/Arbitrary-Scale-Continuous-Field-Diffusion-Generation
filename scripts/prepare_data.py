import argparse
import json
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

MIRRORS = {
    "ffhq": [
        "https://github.com/NVlabs/ffhq-dataset/raw/master/images1024x1024/00000.png"
    ],
    "celebahq": [
        "https://raw.githubusercontent.com/switchablenorms/CelebAMask-HQ/master/CelebA-HQ-img/0.jpg"
    ],
}


def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get('content-length',0))
    with open(path,'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=path.name) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def ensure_structure(root: Path):
    for s in ["train","val"]:
        (root/s).mkdir(parents=True, exist_ok=True)


def prepare(name: str, out_root: Path):
    root = out_root / name
    ensure_structure(root)
    marker = root / "meta.json"
    if marker.exists():
        print(f"{name} already prepared, skip.")
        return

    urls = MIRRORS.get(name.lower(), [])
    ok = False
    for i,u in enumerate(urls):
        try:
            dst = root / "train" / f"sample_{i}.jpg"
            download_file(u, dst)
            shutil.copy(dst, root / "val" / dst.name)
            ok = True
            break
        except Exception as e:
            print(f"Mirror failed: {u}\n  reason={e}")

    if not ok:
        msg = {
            "error": f"Failed auto-download for {name}",
            "manual": f"Please place images under {root}/train and {root}/val. Supported extensions: jpg/png/jpeg/webp"
        }
        raise RuntimeError(json.dumps(msg, indent=2))

    with open(marker,'w',encoding='utf-8') as f:
        json.dump({"dataset":name,"status":"prepared"},f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='ffhq,celebahq')
    ap.add_argument('--data-root', type=str, default='./data')
    args = ap.parse_args()
    for d in [x.strip() for x in args.datasets.split(',') if x.strip()]:
        prepare(d, Path(args.data_root))
    print('Done.')
