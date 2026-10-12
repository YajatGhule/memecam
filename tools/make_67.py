"""Generate assets/sixty_seven.png: a huge bold "67" with a thick outline, transparent background."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/noto/NotoSans-Black.ttf"


def main():
    img = Image.new("RGBA", (900, 620), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 520)
    d.text((450, 300), "67", anchor="mm", font=font, fill=(255, 214, 0), stroke_width=22, stroke_fill="black")
    img = img.crop(img.getbbox())
    img.save(ROOT / "assets" / "sixty_seven.png")
    print("wrote", ROOT / "assets" / "sixty_seven.png")


if __name__ == "__main__":
    main()
