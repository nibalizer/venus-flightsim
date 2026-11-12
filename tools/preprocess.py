"""Convert the raw USGS Magellan GeoTIFFs into compact web assets.

Inputs (data/raw/, both global simple-cylindrical, lon -180..180, lat 90..-90):
  topo.tif  Venus_Magellan_Topography_Global_4641m_v02.tif  int16 metres, nodata -32768
  sar.tif   Venus_Magellan_C3-MDIR_Global_Mosaic_2025m.tif  uint8 radar backscatter, nodata 0

Outputs (data/):
  height.bin.gz   int16 metres, row-wise delta encoded, gzip
  radar_8k.jpg    8192x4096 radar mosaic
  radar_16k.jpg   16384x8192 radar mosaic (used when the GPU supports it)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"


def push_pull_fill(a, valid):
    """Fill invalid pixels with a smooth blend of nearby valid ones (pyramid push-pull)."""
    a = a.astype(np.float32)
    w = valid.astype(np.float32)
    levels = []
    cur_a, cur_w = a * w, w
    while min(cur_a.shape) > 2:
        levels.append((cur_a, cur_w))
        h, wd = cur_a.shape[0] // 2 * 2, cur_a.shape[1] // 2 * 2
        ca, cw = cur_a[:h, :wd], cur_w[:h, :wd]
        cur_a = ca.reshape(h // 2, 2, wd // 2, 2).sum((1, 3))
        cur_w = cw.reshape(h // 2, 2, wd // 2, 2).sum((1, 3))
    filled = cur_a / np.maximum(cur_w, 1e-6)
    for la, lw in reversed(levels):
        up = np.repeat(np.repeat(filled, 2, 0), 2, 1)
        up = np.pad(up, ((0, la.shape[0] - up.shape[0]), (0, la.shape[1] - up.shape[1])), mode="edge")
        mean = la / np.maximum(lw, 1e-6)
        t = np.clip(lw, 0, 1)
        filled = mean * t + up * (1 - t)
    return np.where(valid, a, filled)


def dilate(mask, r):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def box_resize(a, shape):
    img = Image.fromarray(a.astype(np.float32), mode="F")
    return np.asarray(img.resize((shape[1], shape[0]), Image.BOX))


def main():
    print("height...")
    topo = tifffile.imread(RAW / "topo.tif")
    valid = topo != -32768
    print(f"  {topo.shape}, {100 * (1 - valid.mean()):.2f}% gaps")
    topo = push_pull_fill(topo, valid)
    # The DEM's real resolution (~10-20 km altimeter footprint) is well below 4.6 km/px,
    # so halving it loses almost nothing.
    h = np.round(box_resize(topo, (2048, 4096))).astype(np.int16)
    delta = np.diff(h.astype(np.int32), axis=1, prepend=0).astype(np.int16)
    (OUT / "height.bin.gz").write_bytes(gzip.compress(delta.tobytes(), 9))
    meta = {"width": 4096, "height": 2048, "min": int(h.min()), "max": int(h.max())}
    (OUT / "height.json").write_text(json.dumps(meta))
    print(f"  {meta}")

    print("radar...")
    sar = tifffile.imread(RAW / "sar.tif")
    # Gap edges were resampled in the source mosaic, so grow the gaps to swallow the dark fringe.
    valid = ~dilate(sar == 0, 2)
    print(f"  {sar.shape}, {100 * (1 - valid.mean()):.2f}% gaps")
    sar = np.clip(push_pull_fill(sar, valid), 1, 255)
    for name, shape in [("radar_16k.jpg", (8192, 16384)), ("radar_8k.jpg", (4096, 8192))]:
        img = Image.fromarray(np.clip(box_resize(sar, shape), 0, 255).astype(np.uint8), mode="L")
        img.save(OUT / name, quality=82, optimize=True, progressive=True)

    for f in sorted(OUT.glob("*.*")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
