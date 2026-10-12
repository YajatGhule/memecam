"""Generate assets/nerd_finger_up.png: the finger-up nerd with an "umm, actually..." caption.

Uses assets/nerd_source.(jpg|png|webp) if present (the photo of the nerd holding his finger up),
otherwise falls back to a ☝️🤓 built from Twemoji renders.
Usage: uv run python tools/make_nerd.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "assets"
FONT = "/usr/share/fonts/noto/NotoSans-Black.ttf"
CAPTION = "umm, actually..."


def caption(img: Image.Image) -> Image.Image:
    """Classic meme caption: big white text with a black outline along the top."""
    img = img.convert("RGBA")
    w, h = img.size
    d = ImageDraw.Draw(img)
    size = w // 9
    font = ImageFont.truetype(FONT, size)
    while d.textlength(CAPTION, font=font) > w * 0.92:
        size -= 4
        font = ImageFont.truetype(FONT, size)
    d.text((w // 2, int(h * 0.03)), CAPTION, anchor="mt", font=font, fill="white",
           stroke_width=max(3, size // 12), stroke_fill="black")
    return img


def from_photo(src: Path) -> Image.Image:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    # Stock previews carry a branding strip along the bottom: drop it.
    if src.stem == "nerd_source" and h > 900:
        img = img.crop((0, 0, w, int(h * 0.905)))
    img.thumbnail((900, 900), Image.LANCZOS)
    return caption(img)


def from_emoji() -> Image.Image:
    nerd = Image.open(A / "nerd_face.png").convert("RGBA").resize((420, 420), Image.LANCZOS)
    point = Image.open(A / "emoji_point_up.png").convert("RGBA").resize((300, 300), Image.LANCZOS)
    img = Image.new("RGBA", (720, 560), (0, 0, 0, 0))
    img.paste(point, (0, 200), point)
    img.paste(nerd, (270, 120), nerd)
    return caption(img)


def main():
    srcs = [p for ext in ("jpg", "jpeg", "png", "webp") for p in A.glob(f"nerd_source.{ext}")]
    img = from_photo(srcs[0]) if srcs else from_emoji()
    img.save(A / "nerd_finger_up.png")
    print("wrote", A / "nerd_finger_up.png", "from", srcs[0].name if srcs else "emoji")


if __name__ == "__main__":
    main()
