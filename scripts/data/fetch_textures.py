"""Populate the background-texture directory for the PushT OOD cells.

Downloads deterministic natural images (picsum fixed IDs -> static cells)
and extracts frames from imageio's sample videos (-> dynamic clip cells)
into the texture dir resolved the same way the env resolves it:
--out > $SWM_TEXTURE_DIR > swm cache `textures/`.

`background.texture_id` indexes the SORTED entries of this directory, so
entry names are prefixed to pin the ordering:

    01_mountains.jpg   id=1  static natural image
    02_forest.jpg      id=2  static natural image
    03_city.jpg        id=3  static natural image
    04_cockatoo/       id=4  dynamic clip (one frame per env step)
    05_newtonscradle/  id=5  dynamic clip

Usage:
    python scripts/data/fetch_textures.py [--out DIR]
"""

import argparse
import os
import urllib.request
from pathlib import Path

import imageio.v3 as iio

import stable_worldmodel as swm


# stored as PNG: MuJoCo (cube floor texture) only accepts PNG files
IMAGES = {
    '01_mountains.png': 'https://picsum.photos/id/1018/512/512.jpg',
    '02_forest.png': 'https://picsum.photos/id/1015/512/512.jpg',
    '03_city.png': 'https://picsum.photos/id/1011/512/512.jpg',
}

# imageio standard sample videos (downloaded once into imageio's cache)
CLIPS = {
    '04_cockatoo': 'imageio:cockatoo.mp4',
    '05_newtonscradle': 'imageio:newtonscradle.gif',
}

CLIP_MAX_FRAMES = 100

# DAVIS 2017 (480p): the standard natural-video source in the visual-RL
# robustness literature (DMControl-GB etc.; DCS itself uses Kinetics, which
# is YouTube-ID-distributed and not reproducibly fetchable). --davis
# downloads the trainval zip once (~800 MB) and extracts these sequences as
# clip dirs; ids start at 06 so the placeholder ids 1-5 stay stable.
DAVIS_URL = (
    'https://data.vision.ee.ethz.ch/csergi/share/davis/'
    'DAVIS-2017-trainval-480p.zip'
)
DAVIS_SEQUENCES = {
    '06_davis_bear': 'bear',
    '07_davis_dog': 'dog',
    '08_davis_car_roundabout': 'car-roundabout',
}


def fetch_davis(out: Path) -> None:
    import zipfile

    if all((out / name).exists() for name in DAVIS_SEQUENCES):
        print('davis sequences: exist, skipping')
        return

    # keep the zip OUTSIDE the texture dir: every entry inside it shifts the
    # sorted texture_id mapping the envs resolve
    zip_path = out.parent / 'DAVIS-2017-trainval-480p.zip'
    if not zip_path.exists():
        print(f'downloading DAVIS trainval 480p (~800 MB) to {zip_path} ...')
        urllib.request.urlretrieve(DAVIS_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for name, seq in DAVIS_SEQUENCES.items():
            clip_dir = out / name
            if clip_dir.exists():
                print(f'{name}/: exists, skipping')
                continue
            members = sorted(
                m
                for m in zf.namelist()
                if m.startswith(f'DAVIS/JPEGImages/480p/{seq}/')
                and m.endswith('.jpg')
            )[:CLIP_MAX_FRAMES]
            clip_dir.mkdir()
            for t, member in enumerate(members):
                with zf.open(member) as f:
                    frame = iio.imread(f.read(), extension='.jpg')
                iio.imwrite(clip_dir / f'{t:04d}.png', frame)
            print(f'{name}/: {len(members)} frames')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument(
        '--davis',
        action='store_true',
        help='also fetch DAVIS 2017 sequences (one-time ~800 MB download)',
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        env_dir = os.environ.get('SWM_TEXTURE_DIR')
        out = (
            Path(env_dir)
            if env_dir
            else swm.data.utils.get_cache_dir(sub_folder='textures')
        )
    out.mkdir(parents=True, exist_ok=True)

    for name, url in IMAGES.items():
        path = out / name
        if path.exists():
            print(f'{name}: exists, skipping')
            continue
        tmp, _ = urllib.request.urlretrieve(url)
        iio.imwrite(path, iio.imread(tmp))  # re-encode as PNG
        print(f'{name}: downloaded')

    for name, source in CLIPS.items():
        clip_dir = out / name
        if clip_dir.exists():
            print(f'{name}/: exists, skipping')
            continue
        frames = iio.imread(source)
        clip_dir.mkdir()
        for t, frame in enumerate(frames[:CLIP_MAX_FRAMES]):
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            iio.imwrite(clip_dir / f'{t:04d}.png', frame)
        print(f'{name}/: {min(len(frames), CLIP_MAX_FRAMES)} frames')

    if args.davis:
        fetch_davis(out)

    entries = sorted(out.iterdir())
    print(f'\n{out} ready — background.texture_id mapping:')
    for i, entry in enumerate(entries, start=1):
        kind = 'clip' if entry.is_dir() else 'static'
        print(f'  {i}: {entry.name} ({kind})')


if __name__ == '__main__':
    main()
