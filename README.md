# six-seven meme animator

Turn any image into the classic **6/7 wobble meme** — a perspective-warped animation that tilts your image left and right with a gradually slowing rhythm, saved as H.264 MP4 and optionally as an animated GIF.

[100% AI-generated.](https://claude.ai/share/437b5515-44d2-4f16-8936-c5593ec91c02)

---

## how it works

Each full iteration consists of four phases, each lasting `s` seconds:

1. Tilt left (left vertices up, right vertices down by 15% of image height)
2. Return to neutral
3. Tilt right (right vertices up, left vertices down by 15% of image height)
4. Return to neutral

The tilt duration `s` grows across `N` iterations so that the animation slows down over time, reaching `s₀ / d` on the final iteration:

```
s_k = s₀ · r^k     where r = (1/d)^(1/(N−1))
```

All transitions use smooth-step easing (zero velocity at start and end of each phase). Warping is done via OpenCV perspective transform at 60 FPS.

**Defaults:** `s₀ = 0.125 s`, `N = 4`, `d = 0.65`

---

## requirements

- Python 3.10+
- [Pillow](https://pillow.readthedocs.io/)
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- [FFmpeg](https://ffmpeg.org/) (must be on `PATH`)

```bash
pip install pillow opencv-python numpy
```

---

## usage

```
python six_seven_meme.py <input> [options]
```

Output files are written alongside the input, using its basename:

```
photo.jpg  →  photo.mp4  (and optionally photo.gif)
```

Running with no arguments (or `--help`) prints full usage information.

---

## options

| Flag | Description | Default |
|---|---|---|
| `--gif` | Also save an animated GIF alongside the MP4 | off |
| `--bgcolor SPEC` | Background colour revealed during tilt (see below) | `extract` |
| `--height PX` | Output height in pixels (see size rules below) | original size |
| `--s0 SECS` | Base half-swing duration in seconds `[0.05 – 0.5]` | `0.125` |
| `--N INT` | Number of full tilt iterations `[3 – 15]` | `4` |
| `--d FLOAT` | Growth factor — controls how much the animation slows `[0.5 – 1.0]` | `0.65` |

### `--bgcolor` values

| Value | Behaviour |
|---|---|
| `extract` | Most common colour in a 256-palette reduction of the image |
| `average` | Mean RGB value across all pixels |
| `black`, `white`, `grey` | Pure black, white, 50% grey |
| `red`, `green`, `blue` | Primary colours |
| `yellow`, `orange`, `brown`, `pink`, `purple` | Named colours |
| `ff8800` | Any 6-character RGB hex code (with or without `#`) |

### size rules

- The output is **never enlarged** beyond the original image dimensions.
- For images originally taller than 100 px, the minimum output height is **50 px**.
- Neither output dimension may exceed **1080 px**. Oversized inputs are scaled down automatically, preserving aspect ratio.
- Out-of-range `--s0`, `--N`, and `--d` values are clamped with a printed warning rather than rejected.

---

## examples

```bash
# Basic — MP4 only, background colour extracted from image
python six_seven_meme.py photo.jpg

# With GIF, white background
python six_seven_meme.py photo.jpg --gif --bgcolor white

# Custom hex background, resize to 480 px tall
python six_seven_meme.py photo.jpg --bgcolor ff4400 --height 480

# Slower start, more iterations, less slowdown
python six_seven_meme.py photo.jpg --s0 0.2 --N 6 --d 0.8 --gif
```

---

## licence

MIT
