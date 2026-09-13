#!/usr/bin/env python3
# ============================================================
# Main_Text_Figure4_combine_v1.0.py
#   Take Fig4 two little figures into the entire FIG4: 
#     up = a  station violin   (Main_Text_Figure4_station_violin)
#     down = b  air sensitivity  (Main_Text_Figure4_air_sensitivity)
#   作法：讀兩張「已輸出的 PNG」，統一寬度後垂直堆疊（小腳本、與資料無關；
#         先跑完那兩支子圖產生 PNG，再跑本檔）。a / b 字母已各自畫在子圖裡。
#
#   對齊：兩子圖都存成 10in 滿版（violin 本來就是；sensitivity 自 v1.5 起也拿掉了
#   bbox_inches='tight'），所以兩張等寬、左右邊界一致，直接堆疊即可對齊。
#   若日後某張又改成 tight/裁邊而變窄，本檔會把它縮放到同寬，panel 會略微不齊。
# ============================================================
import os
from PIL import Image

FIG_DIR = "/work/home/H.Jason421/water_temp_for_publish/figures/final/"
TOP_PNG = FIG_DIR + "Main_Text_Figure4_station_violin_v1.3_5y_white.png"   # a (up)
BOT_PNG = FIG_DIR + "Main_Text_Figure4_air_sensitivity_v1.6_white.png"     # b (dwon)
OUT_PNG = FIG_DIR + "Main_Text_Figure4_combined_v1.0_white.png"

GUTTER = 40                 # 兩圖之間的白色間隙（px）；要更緊/更鬆改這裡
BG     = (255, 255, 255)    # background (white)
DPI    = 800                # 寫入 PNG 的 dpi 標記（與子圖一致）


def load_rgb(path):
    """讀圖並攤平到白底 RGB（處理透明度）。"""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        base = Image.new("RGB", im.size, BG)
        base.paste(im, mask=im.split()[-1])
        return base
    return im.convert("RGB")


def to_width(im, w):
    """等比例縮放到指定寬度。"""
    if im.width == w:
        return im
    h = round(im.height * w / im.width)
    return im.resize((w, h), Image.LANCZOS)


def main():
    top = load_rgb(TOP_PNG)
    bot = load_rgb(BOT_PNG)

    W = max(top.width, bot.width)          # 統一到較寬者
    top = to_width(top, W)
    bot = to_width(bot, W)

    H = top.height + GUTTER + bot.height
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(top, (0, 0))
    canvas.paste(bot, (0, top.height + GUTTER))

    os.makedirs(FIG_DIR, exist_ok=True)
    canvas.save(OUT_PNG, dpi=(DPI, DPI))
    print(f"Saved: {OUT_PNG}  ({W}x{H}px)")


if __name__ == "__main__":
    main()
