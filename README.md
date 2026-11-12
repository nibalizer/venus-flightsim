# Venus Flight

**Fly it: https://nibalizer.github.io/venus-flightsim/**

In 1989, the Magellan spacecraft took radar readings of the surface of Venus. Since this data is public it was easy to build a flight sim around it.

Very few pictures of the venutian surface exist because the conditions are so intense that the probes landed there fail in minutes or hours. Here are some of the [pictures](https://www.planetary.org/articles/every-picture-from-venus-surface-ever).

A minimal browser flight sim over real Venus, built from NASA Magellan data:

- **Terrain**: Magellan global topography (GTDR, 4.6 km/px), resampled to 4096×2048
- **Surface**: Magellan C3-MDIR synthetic aperture radar mosaic (2 km/px), at 8k or 16k

## Run

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

It needs to be served over HTTP (not opened as a file). Three.js loads from jsDelivr.

## Fly

| Input | |
|---|---|
| Mouse drag | virtual stick (drag down = nose up) |
| Mouse wheel | throttle (10 m/s – 100 km/s) |
| W/S or ↑/↓ | pitch (W = nose down) |
| A/D or ←/→ | roll (banking also turns you) |
| Q/E | yaw |
| Shift / Ctrl | throttle |
| `[` / `]` | vertical exaggeration (default ×3) |
| T | toggle synthetic close-range detail |
| I / P / H / Esc | invert pitch / pause / hide help / about |

On phones and tablets, drag your left thumb to steer and your right thumb up/down for
throttle. Buttons at the top right cycle exaggeration and invert pitch.

Use the "Fly to…" menu or click the minimap to teleport. The URL hash holds your
position (`#lat,lon,alt_km,heading,pitch`), so you can bookmark or share a view.

Altitudes in the HUD are real (unexaggerated). The radar mosaic is ~2 km per pixel, so
it looks best from about 30–300 km up. Down low you're seeing magnified pixels, and the
**T** detail layer (the mosaic tiled at small scale) is synthetic, not real data.

## Data

`data/` holds the processed web assets (~13 MB first load; the 16k radar map adds ~13 MB
when the GPU supports 16384px textures). To rebuild them from the USGS originals:

```sh
mkdir -p data/raw
curl -L -o data/raw/topo.tif https://planetarymaps.usgs.gov/mosaic/Venus_Magellan_Topography_Global_4641m_v02.tif
curl -L -o data/raw/sar.tif  https://planetarymaps.usgs.gov/mosaic/Venus_Magellan_C3-MDIR_Global_Mosaic_2025m.tif
pip install numpy pillow tifffile imagecodecs
python3 tools/preprocess.py
```

The raw files (~245 MB) aren't needed at runtime and can be deleted.

## License

Code: MIT (see `LICENSE`). Three.js (MIT) is loaded from a CDN.

Data: NASA / JPL / USGS Astrogeology, Magellan mission. US public domain; the files in
`data/` are processed derivatives of the USGS mosaics linked above.
