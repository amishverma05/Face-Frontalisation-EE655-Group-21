"""
download_data.py
─────────────────
Helper script to download and verify the 300W-LP and AFLW2000-3D datasets.

300W-LP
━━━━━━━
  Official site: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
  Mirror (Google Drive): https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view
  Size: ~2.7 GB

AFLW2000-3D
━━━━━━━━━━━
  Official: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
  Size: ~200 MB

This script:
  1. Tries to auto-download via gdown (Google Drive)
  2. If unavailable, prints manual download instructions
  3. Validates dataset structure after download
"""

import os
import sys
import subprocess
import zipfile
import glob
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Google Drive IDs  (update if links expire)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: These are community mirrors; official requires registration.
URLS = {
    '300W_LP': {
        'gdrive_id': '0B7OEHD3T4eCkVGs0TkhUWFN6N1k',
        'filename':  '300W-LP.zip',
        'dest_dir':  'data/300W_LP',
    },
    'AFLW2000': {
        'gdrive_id': '1Nf4Z-XxJCx4bMEXt0MXCY5G8_E9TnC2',
        'filename':  'AFLW2000-3D.zip',
        'dest_dir':  'data/AFLW2000',
    },
}


def check_gdown():
    try:
        import gdown
        return True
    except ImportError:
        return False


def install_gdown():
    print("Installing gdown ...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown', '-q'])


def download_gdrive(file_id: str, output: str):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)


def extract_zip(zip_path: str, dest: str):
    print(f"Extracting {zip_path} → {dest} ...")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)
    print("  Done.")


def validate_dataset(root: str, min_files: int = 100) -> bool:
    """Check that dataset has at least `min_files` .jpg + .mat pairs."""
    jpgs = glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True)
    mats = glob.glob(os.path.join(root, '**', '*.mat'), recursive=True)
    ok   = len(jpgs) >= min_files and len(mats) >= min_files
    print(f"  [{root}]  {len(jpgs):,} images | {len(mats):,} mat files  →  {'✅ OK' if ok else '❌ INCOMPLETE'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────

def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            MANUAL DOWNLOAD INSTRUCTIONS                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/           ║
║            projects/3DDFA/main.htm                               ║
║                                                                  ║
║  2. Download:                                                    ║
║     • 300W-LP.zip    (~2.7 GB)                                  ║
║     • AFLW2000-3D.zip (~200 MB)                                  ║
║                                                                  ║
║  3. Place them in this project root, then run:                   ║
║     python download_data.py --extract-only                       ║
║                                                                  ║
║  Alternative mirrors:                                            ║
║  • https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k║
║    (300W-LP, requires Google account)                            ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download face frontalization datasets')
    parser.add_argument('--extract-only', action='store_true',
                        help='Skip download, just extract existing zips')
    parser.add_argument('--validate-only', action='store_true',
                        help='Skip download/extract, just validate existing data')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    if args.validate_only:
        print("\nValidating datasets ...\n")
        for name, info in URLS.items():
            validate_dataset(info['dest_dir'])
        return

    if args.extract_only:
        for name, info in URLS.items():
            zip_path = info['filename']
            if os.path.exists(zip_path):
                extract_zip(zip_path, info['dest_dir'])
            else:
                print(f"  ⚠ {zip_path} not found. Skipping.")
        return

    # Auto-download attempt
    if not check_gdown():
        install_gdown()

    success = True
    for name, info in URLS.items():
        dest = info['dest_dir']
        if validate_dataset(dest, min_files=100):
            print(f"  ✅ {name} already present, skipping download.")
            continue

        zip_path = info['filename']
        if not os.path.exists(zip_path):
            print(f"\nDownloading {name} ...")
            try:
                download_gdrive(info['gdrive_id'], zip_path)
            except Exception as e:
                print(f"  ❌ Auto-download failed: {e}")
                success = False
                continue

        extract_zip(zip_path, dest)

    if not success:
        print_manual_instructions()

    print("\nFinal validation:")
    for name, info in URLS.items():
        validate_dataset(info['dest_dir'])


if __name__ == '__main__':
    main()
