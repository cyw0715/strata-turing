"""Create deterministic, non-private image fixtures for the live image API."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

folder = Path(__file__).resolve().parent / 'vision-smoke'
folder.mkdir(exist_ok=True)
import os
font = ImageFont.truetype(os.environ.get('STRATA_TEST_FONT', 'DejaVuSans-Bold.ttf'), 48)
for name, label in [('a', 'TEST 4729'), ('b', 'TEST 8361')]:
    image = Image.new('RGB', (512, 384), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((95, 25), label, font=font, fill='black')
    if name == 'a':
        draw.ellipse((35, 145, 215, 325), fill=(0, 70, 240))
        draw.rectangle((295, 145, 475, 325), fill=(240, 20, 20))
    else:
        draw.polygon([(125, 135), (25, 325), (225, 325)], fill=(0, 155, 40))
        draw.ellipse((295, 145, 475, 325), fill=(250, 210, 0))
    image.save(folder / f'{name}.png')
    print(f'{name}.png: {image.width}x{image.height}')
