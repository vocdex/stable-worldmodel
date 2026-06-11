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


IMAGES = {
    '01_mountains.jpg': 'https://picsum.photos/id/1018/512/512.jpg',
    '02_forest.jpg': 'https://picsum.photos/id/1015/512/512.jpg',
    '03_city.jpg': 'https://picsum.photos/id/1011/512/512.jpg',
}

# imageio standard sample videos (downloaded once into imageio's cache)
CLIPS = {
    '04_cockatoo': 'imageio:cockatoo.mp4',
    '05_newtonscradle': 'imageio:newtonscradle.gif',
}

CLIP_MAX_FRAMES = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=None)
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
        urllib.request.urlretrieve(url, path)
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

    entries = sorted(out.iterdir())
    print(f'\n{out} ready — background.texture_id mapping:')
    for i, entry in enumerate(entries, start=1):
        kind = 'clip' if entry.is_dir() else 'static'
        print(f'  {i}: {entry.name} ({kind})')


if __name__ == '__main__':
    main()
