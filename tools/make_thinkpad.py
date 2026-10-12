"""Generate assets/thinkpad_middle_finger.png: a ThinkPad flipping you off back.

Takes a ThinkPad product shot, perspective-warps the 🖕 emoji (assets/emoji_middle_finger.png) onto its screen.
Usage: uv run python tools/make_thinkpad.py <thinkpad_frame.png>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
EMOJI_PNG = ROOT / "assets" / "emoji_middle_finger.png"  # Twemoji 1f595, rasterized at 512px
FONT = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
# Screen corners in the 498x281 source frame: TL, TR, BR, BL
SCREEN = np.float32([[193, 52], [371, 69], [353, 177], [181, 158]])


def screen_image(w=640, h=400):
    img = Image.new("RGB", (w, h), (12, 12, 16))
    d = ImageDraw.Draw(img)
    e = Image.open(EMOJI_PNG).convert("RGBA").resize((290, 290), Image.LANCZOS)
    img.paste(e, ((w - 290) // 2, 20), e)
    d.text((w // 2, h - 42), "no u", anchor="mm", fill=(230, 30, 40), font=ImageFont.truetype(FONT, 56))
    return img


def main(src):
    base = cv2.cvtColor(np.array(Image.open(src).convert("RGB")), cv2.COLOR_RGB2BGR)
    base = cv2.resize(base, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # warp at 2x so text stays crisp
    scr = cv2.cvtColor(np.array(screen_image()), cv2.COLOR_RGB2BGR)
    h, w = scr.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), SCREEN * 2)
    warped = cv2.warpPerspective(scr, M, (base.shape[1], base.shape[0]))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (base.shape[1], base.shape[0]))
    base[mask > 0] = warped[mask > 0]
    base = base[92:, 180:840]  # drop the marketing headline, centre the laptop
    out = ROOT / "assets" / "thinkpad_middle_finger.png"
    cv2.imwrite(str(out), base)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
