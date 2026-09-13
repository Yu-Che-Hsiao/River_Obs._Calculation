#!/usr/bin/env python3
# ============================================================
# Supp_figure4_combine.py
#   Stack the two subplots of the Supplementary version of Fig. 4 vertically
#     up = a  station violin   (Supp_figure4_station_violin.py)
#     bottom = b  air sensitivity  (Supp_figure4_air_sensitivity.py)
#
#   Usage (run the two scripts above first)
#     python Supp_figure4_combine.py --min_years 8
#     python Supp_figure4_combine.py --min_years 10
#
#   Both subplots are saved at full 10in width, uncropped, so they are equal width with matching left/right margins and stack into alignment directly.
#   The a / b letters are already drawn inside each subplot.
#
#   Output -> figures/Supplementary_correct/Supplementary_Fig4_combined_{ny}y.png
# ============================================================
import os
import argparse
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--min_years', type=int, default=8)
MIN_YEARS = ap.parse_args().min_years

FIG_DIR = "/work/home/H.Jason421/water_temp_for_publish/figures/Supplementary_correct/"
TOP_PNG = FIG_DIR + f"Supplementary_Fig4a_station_violin_{MIN_YEARS}y.png"
BOT_PNG = FIG_DIR + f"Supplementary_Fig4b_air_sensitivity_{MIN_YEARS}y.png"
OUT_PNG = FIG_DIR + f"Supplementary_Fig4_combined_{MIN_YEARS}y.png"

GUTTER = 40                 # white gap between the two figures (px)
BG     = (255, 255, 255)
DPI    = 800


def load_rgb(path):
    """Read an image and flatten onto a white background as RGB (handles transparency)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        base = Image.new("RGB", im.size, BG)
        base.paste(im, mask=im.split()[-1])
        return base
    return im.convert("RGB")


def to_width(im, w):
    if im.width == w:
        return im
    h = round(im.height * w / im.width)
    return im.resize((w, h), Image.LANCZOS)


def main():
    missing = [p for p in (TOP_PNG, BOT_PNG) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            'Missing subplot(s):\n  ' + '\n  '.join(os.path.basename(p) for p in missing) +
            f'\n\nRun these first:\n'
            f'  python Supp_figure4_station_violin.py   --min_years {MIN_YEARS}\n'
            f'  python Supp_figure4_air_sensitivity.py  --min_years {MIN_YEARS}')

    top = load_rgb(TOP_PNG)
    bot = load_rgb(BOT_PNG)

    W = max(top.width, bot.width)
    if top.width != bot.width:
        print(f'  [Note] the two subplots have different widths ({top.width} vs {bot.width})'
              f'rescaled to {W}; panels may be slightly misaligned.')
    top = to_width(top, W)
    bot = to_width(bot, W)

    H = top.height + GUTTER + bot.height
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(top, (0, 0))
    canvas.paste(bot, (0, top.height + GUTTER))

    canvas.save(OUT_PNG, dpi=(DPI, DPI))
    print(f"Saved: {OUT_PNG}  ({W}x{H}px)")


if __name__ == "__main__":
    main()
