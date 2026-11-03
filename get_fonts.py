"""Download the handwriting fonts used for synthetic signatures (Google Fonts, open licence).

Usage: python get_fonts.py [--out fonts]
"""
import argparse, os, urllib.request

FONTS = [
    "ofl/greatvibes/GreatVibes-Regular.ttf", "ofl/allura/Allura-Regular.ttf",
    "ofl/sacramento/Sacramento-Regular.ttf", "apache/homemadeapple/HomemadeApple-Regular.ttf",
    "ofl/alexbrush/AlexBrush-Regular.ttf", "ofl/parisienne/Parisienne-Regular.ttf",
    "ofl/pinyonscript/PinyonScript-Regular.ttf", "ofl/mrdafoe/MrDafoe-Regular.ttf",
    "ofl/herrvonmuellerhoff/HerrVonMuellerhoff-Regular.ttf", "ofl/italianno/Italianno-Regular.ttf",
    "apache/yellowtail/Yellowtail-Regular.ttf", "ofl/kristi/Kristi-Regular.ttf",
    "ofl/labelleaurore/LaBelleAurore.ttf", "ofl/reeniebeanie/ReenieBeanie.ttf",
    "ofl/monsieurladoulaise/MonsieurLaDoulaise-Regular.ttf", "ofl/marckscript/MarckScript-Regular.ttf",
    "ofl/meddon/Meddon.ttf", "ofl/qwigley/Qwigley-Regular.ttf",
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="fonts")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

for f in FONTS:
    dst = os.path.join(args.out, os.path.basename(f))
    if os.path.exists(dst):
        continue
    try:
        urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/" + f, dst)
    except Exception as e:
        print("font not found (skipped):", f, e)
print(len(os.listdir(args.out)), "fonts ready in", args.out)
