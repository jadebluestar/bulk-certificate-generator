import os
import json
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 1100
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

for i in range(12):
    d.rectangle([20 + i, 20 + i, W - 20 - i, H - 20 - i],
                outline=(180, 150, 60) if i % 2 == 0 else (120, 90, 30))

def font(size):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# Track every placeholder we render: name -> (x, y, w, h)
placements = {}

def center_placeholder(text, y, f, field, fill=(20, 20, 20)):
    bbox = d.textbbox((0, 0), text, font=f)      # bbox = (x0, y0, x1, y1)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # draw so the visual top of the glyphs lands at y
    x = (W - tw) // 2 - bbox[0]
    ty = y - bbox[1]
    d.text((x, ty), text, font=f, fill=fill)
    # record the ACTUAL drawn bounds (matches what the renderer will erase)
    placements[field] = [
        int(x + bbox[0]),
        int(ty + bbox[1]),
        int(tw),
        int(th),
    ]

def center(text, y, f, fill=(30, 30, 30)):
    bbox = d.textbbox((0, 0), text, font=f)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) / 2, y), text, font=f, fill=fill)

# Title stuff (no placeholders)
center("CERTIFICATE", 120, font(90), (150, 110, 20))
center("OF ACHIEVEMENT", 230, font(46), (150, 110, 20))
center("This certificate is proudly presented to", 380, font(34), (80, 80, 80))

# Placeholders — record their exact boxes
center_placeholder("{{name}}",   470, font(80), "name")
center("for successfully completing the course", 640, font(30), (80, 80, 80))
center_placeholder("{{course}}", 700, font(46), "course")
center_placeholder("Date: {{date}}", 880, font(30), "date")
center("Authorized Signature", 980, font(28), (80, 80, 80))

os.makedirs("templates", exist_ok=True)
img.save("templates/certificate_template.png")

with open("templates/certificate_template.placements.json", "w") as fh:
    json.dump(placements, fh, indent=2)

print("Template saved.")
print("Placements:", json.dumps(placements, indent=2))