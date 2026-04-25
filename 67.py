"""
Six-Seven Meme Animator
Turns any image into the "6/7" wobble meme animation.

Parameters:
  s0 = 0.125   — base duration of each half-swing (seconds)
  N  = 4       — number of full iterations (left-tilt + right-tilt)
  d  = 0.65    — growth factor; s in iteration k = s0 * r^(k-1)
                 where r solves r^(N-1) = 1/d  →  r = (1/d)^(1/(N-1))
"""

import sys
import argparse
import numpy as np
from PIL import Image
import subprocess
import os
import tempfile
import shutil

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_S0 = 0.125
DEFAULT_N  = 4
DEFAULT_D  = 0.65
FPS        = 60
LIFT_FRAC  = 0.15   # 15% of image height

MAX_DIM    = 1080   # hard ceiling on output width or height
MIN_HEIGHT = 50     # minimum output height (for images originally > 100 px tall)

# ── Named background colours ──────────────────────────────────────────────────
NAMED_COLORS = {
    "black":  (0,   0,   0),
    "white":  (255, 255, 255),
    "grey":   (128, 128, 128),
    "gray":   (128, 128, 128),
    "red":    (255, 0,   0),
    "green":  (0,   128, 0),
    "blue":   (0,   0,   255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "brown":  (139, 69,  19),
    "pink":   (255, 192, 203),
    "purple": (128, 0,   128),
}

# ── Colour helpers ────────────────────────────────────────────────────────────
def parse_bgcolor(spec: str, img: Image.Image) -> tuple:
    """Resolve a --bgcolor spec to an (R, G, B) tuple."""
    s = spec.strip().lower()

    if s in NAMED_COLORS:
        return NAMED_COLORS[s]

    if s == "average":
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        avg = arr.mean(axis=(0, 1))
        return (int(round(avg[0])), int(round(avg[1])), int(round(avg[2])))

    if s == "extract":
        # Reduce to 256-colour palette and pick most common colour
        paletted = img.convert("RGB").quantize(colors=256)
        palette_data = paletted.getpalette()          # flat list R G B R G B ...
        counts = paletted.getcolors(maxcolors=256)    # [(count, idx), ...]
        if not counts:
            return (0, 0, 0)
        most_common_idx = max(counts, key=lambda x: x[0])[1]
        r = palette_data[most_common_idx * 3]
        g = palette_data[most_common_idx * 3 + 1]
        b = palette_data[most_common_idx * 3 + 2]
        return (r, g, b)

    # Try 6-character hex code (with or without leading #)
    hex_str = s.lstrip("#")
    if len(hex_str) == 6:
        try:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (r, g, b)
        except ValueError:
            pass

    raise ValueError(
        f"Unrecognised --bgcolor value: '{spec}'\n"
        f"Valid options: {', '.join(k for k in NAMED_COLORS if k != 'gray')}, "
        f"extract, average, or a 6-character hex code (e.g. ff8800)."
    )


# ── Size helpers ──────────────────────────────────────────────────────────────
def compute_output_size(orig_w: int, orig_h: int, requested_h) -> tuple:
    """
    Return (out_w, out_h) respecting all constraints:
      - Never enlarge beyond original dimensions.
      - For images originally > 100 px tall, minimum output height is 50 px.
      - Hard ceiling: neither dimension may exceed MAX_DIM (1080).
      - Aspect ratio is preserved.
    """
    target_h = requested_h if requested_h is not None else orig_h

    # Never enlarge
    target_h = min(target_h, orig_h)

    # Apply minimum (only for images originally taller than 100 px)
    if orig_h > 100:
        target_h = max(target_h, MIN_HEIGHT)

    # Scale width proportionally
    scale = target_h / orig_h
    target_w = int(round(orig_w * scale))

    # Enforce MAX_DIM ceiling — recalculate if either axis is over
    if target_h > MAX_DIM or target_w > MAX_DIM:
        scale = min(MAX_DIM / target_h, MAX_DIM / target_w)
        target_h = int(round(target_h * scale))
        target_w = int(round(target_w * scale))

    # Also cap if original itself was oversized (no --height given)
    if orig_h > MAX_DIM or orig_w > MAX_DIM:
        cap_scale = min(MAX_DIM / orig_h, MAX_DIM / orig_w)
        cap_h = int(round(orig_h * cap_scale))
        cap_w = int(round(orig_w * cap_scale))
        if target_h > cap_h:
            target_h = cap_h
            target_w = cap_w

    return (target_w, target_h)


# ── Easing ────────────────────────────────────────────────────────────────────
def ease_in_out(t: float) -> float:
    """Smooth-step: 0→1 with zero velocity at both ends."""
    return t * t * (3 - 2 * t)


# ── Perspective warp ──────────────────────────────────────────────────────────
def warp_image(img_np, src_pts, dst_pts,
               canvas_h: int, canvas_w: int, origin_y: int,
               bg_color: tuple):
    """
    Perspective-warp img_np from src_pts -> dst_pts onto a canvas pre-filled
    with bg_color, then composite.
    """
    import cv2

    # Canvas pre-filled with background colour (fully opaque)
    canvas = np.empty((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas[:, :, 0] = bg_color[0]
    canvas[:, :, 1] = bg_color[1]
    canvas[:, :, 2] = bg_color[2]
    canvas[:, :, 3] = 255

    dst_canvas = dst_pts.copy()
    dst_canvas[:, 1] += origin_y

    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32),
                                    dst_canvas.astype(np.float32))

    # Warp with BORDER_TRANSPARENT so uncovered canvas pixels stay as background
    warped = cv2.warpPerspective(img_np, M, (canvas_w, canvas_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=canvas.copy())

    return warped


def make_frame(img_np, lift: float,
               canvas_h: int, origin_y: int,
               bg_color: tuple) -> Image.Image:
    """
    lift > 0  left side up, right side down
    lift < 0  right side up, left side down
    |lift| in pixels
    """
    h, w = img_np.shape[:2]

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = np.array([[0, -lift], [w, lift], [w, h + lift], [0, h - lift]],
                   dtype=np.float32)

    canvas_arr = warp_image(img_np, src, dst, canvas_h, w, origin_y, bg_color)
    crop = canvas_arr[origin_y: origin_y + h, :, :]
    return Image.fromarray(crop, mode='RGBA')


# ── Iteration schedule ────────────────────────────────────────────────────────
def iter_durations(s0: float, n: int, d: float) -> list:
    """
    s grows so that s_N = s0 / d.
    s_k = s0 * r^k  where r^(N-1) = 1/d  ->  r = (1/d)^(1/(N-1))
    """
    if n == 1:
        return [s0]
    r = (1.0 / d) ** (1.0 / (n - 1))
    return [s0 * (r ** k) for k in range(n)]


# ── Main frame builder ────────────────────────────────────────────────────────
def build_frames(img: Image.Image,
                 bg_color: tuple,
                 s0: float, n: int, d: float) -> list:
    img_rgba = img.convert('RGBA')
    img_np = np.array(img_rgba)

    h, w = img_np.shape[:2]
    max_lift = int(round(LIFT_FRAC * h))

    pad = max_lift + 4
    canvas_h = h + 2 * pad
    origin_y = pad

    durations = iter_durations(s0, n, d)
    frames = []

    def add_phase(lift_sign: int, s: float):
        nf = max(1, int(round(s * FPS)))
        for i in range(nf):
            t = ease_in_out((i + 1) / nf)
            frames.append(make_frame(img_np, lift_sign * max_lift * t,
                                     canvas_h, origin_y, bg_color))
        for i in range(nf):
            t = ease_in_out((i + 1) / nf)
            frames.append(make_frame(img_np, lift_sign * max_lift * (1 - t),
                                     canvas_h, origin_y, bg_color))

    frames.append(make_frame(img_np, 0.0, canvas_h, origin_y, bg_color))
    for s in durations:
        add_phase(+1, s)
        add_phase(-1, s)
    frames.append(make_frame(img_np, 0.0, canvas_h, origin_y, bg_color))

    return frames


# ── Savers ────────────────────────────────────────────────────────────────────
def save_gif(frames: list, path: str):
    rgba_frames = [f.convert("RGBA") for f in frames]
    rgba_frames[0].save(
        path,
        save_all=True,
        append_images=rgba_frames[1:],
        loop=0,
        duration=[int(1000 / FPS)] * len(frames),
        disposal=2,
    )
    print(f"GIF saved  -> {path}")


def save_mp4(frames: list, path: str, tmp_dir: str):
    frame_pattern = os.path.join(tmp_dir, 'frame_%05d.png')
    for i, f in enumerate(frames):
        f.convert('RGB').save(os.path.join(tmp_dir, f'frame_{i:05d}.png'))

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-preset', 'slow',
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr)
    else:
        print(f"MP4 saved  -> {path}")


# ── Argument validation ───────────────────────────────────────────────────────
def clamp_warn(val, lo, hi, name):
    if val < lo:
        print(f"Warning: {name}={val} is below minimum {lo}; clamping to {lo}.")
        return lo
    if val > hi:
        print(f"Warning: {name}={val} is above maximum {hi}; clamping to {hi}.")
        return hi
    return val


# ── Help epilog ───────────────────────────────────────────────────────────────
HELP_EPILOG = """
arguments & rules
-----------------
  input              Path to any image PIL can open (PNG, JPEG, WEBP, BMP, ...).
                     Required. If omitted this help text is shown.

  --gif              Also save an animated GIF alongside the MP4.
                     Off by default (MP4 only).

  --bgcolor SPEC     Background colour revealed when vertices shift inward.
                     Default: extract

                     Named colours : black, white, grey, red, green, blue,
                                     yellow, orange, brown, pink, purple
                     Special       : extract  -- most common colour in a
                                                 256-palette reduction of the image
                                   : average  -- mean RGB of all pixels
                     Hex code      : 6-character RGB hex, e.g.  ff8800 or #ff8800

  --height PX        Output height in pixels.
                     * Never enlarges beyond the original image height.
                     * For images originally > 100 px tall, minimum is 50 px.
                     * Hard ceiling: 1080 px on both width and height.
                     Default: same as original (ceiling limits still apply).

  --s0 SECS          Base half-swing duration in seconds.
                     Range: 0.05 - 0.5    Default: 0.125

  --N INT            Number of full tilt iterations (left + right = 1 iteration).
                     Range: 3 - 15        Default: 4

  --d FLOAT          Growth factor. Controls how much each swing slows relative
                     to the previous one. The Nth swing lasts s0/d seconds.
                     Range: 0.5 - 1.0     Default: 0.65

examples
--------
  python six_seven_meme.py photo.jpg
  python six_seven_meme.py photo.jpg --gif --bgcolor white
  python six_seven_meme.py photo.jpg --bgcolor ff4400 --height 480
  python six_seven_meme.py photo.jpg --s0 0.2 --N 6 --d 0.8 --gif
"""


# ── CLI entry point ───────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="six_seven_meme.py",
        description="Turn any image into the six-seven wobble meme animation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
        add_help=True,
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Input image file.",
    )
    parser.add_argument(
        "--gif", action="store_true", default=False,
        help="Also output an animated GIF (off by default).",
    )
    parser.add_argument(
        "--bgcolor", default="extract", metavar="SPEC",
        help="Background colour (see rules below). Default: extract",
    )
    parser.add_argument(
        "--height", type=int, default=None, metavar="PX",
        help="Output height in pixels (see rules below).",
    )
    parser.add_argument(
        "--s0", type=float, default=DEFAULT_S0, metavar="SECS",
        help=f"Base half-swing duration in seconds [0.05-0.5]. Default: {DEFAULT_S0}",
    )
    parser.add_argument(
        "--N", type=int, default=DEFAULT_N, metavar="INT",
        help=f"Number of tilt iterations [3-15]. Default: {DEFAULT_N}",
    )
    parser.add_argument(
        "--d", type=float, default=DEFAULT_D, metavar="FLOAT",
        help=f"Growth factor [0.5-1.0]. Default: {DEFAULT_D}",
    )
    return parser


def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(0)

    # Validate / clamp animation parameters
    s0 = clamp_warn(args.s0, 0.05, 0.5,  "--s0")
    n  = clamp_warn(args.N,  3,    15,   "--N")
    d  = clamp_warn(args.d,  0.5,  1.0,  "--d")

    # Load image
    print(f"Loading '{args.input}' ...")
    try:
        img = Image.open(args.input)
        img.load()
    except Exception as e:
        sys.exit(f"Error opening image: {e}")

    orig_w, orig_h = img.size

    # Resolve background colour
    try:
        bg_color = parse_bgcolor(args.bgcolor, img)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Background colour: {args.bgcolor!r} -> RGB{bg_color}")

    # Compute output size
    out_w, out_h = compute_output_size(orig_w, orig_h, args.height)
    if (out_w, out_h) != (orig_w, orig_h):
        print(f"Resizing: {orig_w}x{orig_h} -> {out_w}x{out_h}")
        img = img.resize((out_w, out_h), Image.LANCZOS)
    else:
        print(f"Image size: {out_w}x{out_h} (no resize needed)")

    # Derive output paths from input basename
    base = os.path.splitext(args.input)[0]
    gif_path = base + ".gif"
    mp4_path = base + ".mp4"

    # Build frames
    print(f"Building frames (s0={s0}, N={n}, d={d}, FPS={FPS}) ...")
    frames = build_frames(img, bg_color, s0, n, d)
    print(f"  {len(frames)} frames generated.")

    # Save outputs
    tmp_dir = tempfile.mkdtemp(prefix='six_seven_')
    try:
        if args.gif:
            save_gif(frames, gif_path)
        save_mp4(frames, mp4_path, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
