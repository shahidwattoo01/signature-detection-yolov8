"""Build a synthetic YOLO dataset for signature detection.

Every document page gets:
  - 0-3 signatures (labelled, class 0) from three sources: CEDAR scans,
    handwriting fonts and random pen strokes (flowing, zigzag, looped)
  - distractors that are NOT labelled: rubber stamps, empty signature lines,
    tick marks
  - scan-like degradation (blur, noise, contrast, JPEG)

Validation images use CEDAR writers and fonts never seen in training.

Usage:
  python make_dataset.py --pages pages --signatures signatures --fonts fonts --out dataset
"""
import argparse, glob, os, random, zlib
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# BGR
INK_COLORS = [(20, 20, 20), (15, 15, 60), (140, 40, 20), (170, 70, 30),
              (30, 30, 150), (40, 110, 40), (110, 40, 110)]
STAMP_COLORS = [(40, 40, 190), (160, 60, 30), (50, 130, 50), (120, 50, 120), (45, 45, 45)]

STAMP_WORDS = ["RECEIVED", "PAID", "APPROVED", "VERIFIED", "CONFIDENTIAL", "CERTIFIED",
               "WITNESSED", "FILED", "COPY", "ORIGINAL", "URGENT", "DRAFT", "CANCELLED",
               "REJECTED", "ENTERED", "POSTED", "SCANNED", "CHECKED", "VOID", "FINAL",
               "ACCEPTED", "PROCESSED", "COMPLETED", "DUPLICATE", "SEEN", "NOTARIZED"]
SMALL_WORDS = ["ACCOUNTS DEPT", "HEAD OFFICE", "LEGAL DEPT", "ACME CORP", "FINANCE",
               "ADMIN OFFICE", "HR DEPARTMENT", "QUALITY CONTROL", "REGISTRY", "MAIN BRANCH",
               "RECORDS", "CITY COURT", "PAYROLL", "OFFICIAL"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SIGN_LABELS = ["Signature:", "Signed:", "Authorized Signature", "Signature of Employee",
               "Witness:", "Approved by:", "Client Signature", "Sign here", "Signature / Date",
               "Signature of Applicant", "For and on behalf of", ""]
FIRST = ["James", "Mary", "John", "Linda", "Ahmed", "Fatima", "Wei", "Priya", "Carlos", "Anna",
         "David", "Sara", "Omar", "Elena", "Raj", "Grace", "Lucas", "Aisha", "Peter", "Mei",
         "Ali", "Nina", "Hassan", "Olivia", "Ivan", "Zara", "Kenji", "Laura", "Tom", "Sofia"]
LAST = ["Smith", "Khan", "Garcia", "Chen", "Patel", "Brown", "Raza", "Muller", "Rossi", "Kim",
        "Johnson", "Ali", "Silva", "Novak", "Haddad", "Wilson", "Tanaka", "Lopez", "Ahmed",
        "Clarke", "Ivanova", "Singh", "Martin", "Okafor", "Fischer", "Dubois", "Nguyen"]
HERSHEY = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
           cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX]

SIGS, FONTS = {}, {}   # filled per worker by init()


def writer_id(path):
    """CEDAR files look like original_12_3.png; the middle number is the writer."""
    parts = os.path.basename(path).split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1])
    return zlib.crc32(os.path.basename(path).encode())


def init(sig_dir, font_dir):
    sigs = sorted(p for ext in ("png", "jpg", "jpeg", "tif", "bmp")
                  for p in glob.glob(f"{sig_dir}/**/*.{ext}", recursive=True))
    SIGS["val"] = [p for p in sigs if writer_id(p) % 10 == 0]
    SIGS["train"] = [p for p in sigs if writer_id(p) % 10 != 0]
    if not SIGS["val"]:
        SIGS["val"] = SIGS["train"]

    fonts = sorted(glob.glob(f"{font_dir}/*.ttf") + glob.glob(f"{font_dir}/*.otf")) if font_dir else []
    FONTS["val"] = fonts[::6] if len(fonts) >= 3 else fonts
    FONTS["train"] = [f for f in fonts if f not in FONTS["val"]] or fonts


# ---------------------------------------------------------------- mask helpers

def crop(mask):
    ys, xs = np.where(mask > 30)
    if len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def rotate(mask, deg):
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(mask, M, (nw, nh))


def wear(mask, keep):
    """Knock out random blotches of ink, like a dry stamp or faded pen."""
    h, w = mask.shape
    small = np.random.rand(max(2, h // 8), max(2, w // 8)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return (mask * (field < keep)).astype(np.uint8)


def chaikin(pts, iters):
    for _ in range(iters):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.vstack([pts[:1], np.stack([q, r], 1).reshape(-1, 2), pts[-1:]])
    return pts


# ---------------------------------------------------------------- signatures

def sig_cedar(subset):
    img = cv2.imread(random.choice(SIGS[subset]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # drop specks
    return crop(cv2.GaussianBlur(ink, (3, 3), 0))


def sig_font(subset):
    if not FONTS.get(subset):
        return None
    first, last = random.choice(FIRST), random.choice(LAST)
    text = random.choice([f"{first} {last}", f"{first[0]}. {last}", last, f"{first} {last[0]}.",
                          f"{first[0]}{last[0]}", f"{first[0]}. {last[0]}.", first])
    try:
        font = ImageFont.truetype(random.choice(FONTS[subset]), random.randint(60, 120))
    except OSError:
        return None
    x0, y0, x1, y1 = font.getbbox(text)
    pad = 30
    img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad + 40), 0)
    d = ImageDraw.Draw(img)
    d.text((pad - x0, pad - y0), text, font=font, fill=255,
           stroke_width=random.choice([0, 0, 0, 1]), stroke_fill=255)
    mask = np.array(img)
    if random.random() < 0.35:   # underline flourish
        h, w = mask.shape
        y = h - pad - random.randint(0, 25)
        pts = chaikin(np.array([[pad, y], [w * random.uniform(0.3, 0.7), y + random.randint(-15, 15)],
                                [w - pad, y - random.randint(0, 25)]], np.float32), 3)
        cv2.polylines(mask, [pts.astype(np.int32)], False, 255, random.randint(2, 4), cv2.LINE_AA)
    return crop(mask)


def sig_scribble():
    style = random.choice(["flow", "zigzag", "loop", "loop"])
    if style == "loop":   # trochoid: cursive-like loops
        t = np.linspace(0, random.uniform(4, 12) * np.pi, 400)
        r = random.uniform(15, 35) * (1 + 0.4 * np.sin(t * random.uniform(0.2, 0.6)))
        a = random.uniform(5, 14)
        pts = np.stack([a * t - r * np.sin(t), -r * np.cos(t) * random.uniform(0.8, 1.6)], 1)
    else:
        n = random.randint(6, 26)
        x = np.cumsum(np.random.uniform(8, 30, n))
        if style == "zigzag":
            y = np.array([(-1) ** i for i in range(n)]) * np.random.uniform(15, 45, n) + np.random.normal(0, 5, n)
        else:
            y = np.cumsum(np.random.normal(0, 18, n))
            y = (y - y.mean()) * 0.5 + random.uniform(15, 45) * np.sin(np.arange(n) * random.uniform(0.8, 2.5))
        pts = np.stack([x, y], 1)
        if style == "flow":
            pts = chaikin(pts, 3)
    pts = pts - pts.min(0) + 30
    w, h = int(pts[:, 0].max()) + 30, int(pts[:, 1].max()) + 60
    mask = np.zeros((h, w), np.uint8)
    thick = random.randint(2, 5)
    cv2.polylines(mask, [pts.astype(np.int32)], False, 255, thick, cv2.LINE_AA)
    if random.random() < 0.4:   # underline or crossing stroke
        y = int(random.uniform(0.6, 1.0) * (h - 40))
        line = chaikin(np.array([[random.uniform(0, 0.3) * w, y],
                                 [w * 0.5, y + random.randint(-12, 12)],
                                 [random.uniform(0.7, 1.0) * w, y - random.randint(0, 20)]], np.float32), 3)
        cv2.polylines(mask, [line.astype(np.int32)], False, 255, thick, cv2.LINE_AA)
    return crop(mask)


def make_signature(subset):
    r = random.random()
    m = sig_cedar(subset) if r < 0.45 else sig_font(subset) if r < 0.75 else None
    return m if m is not None else sig_scribble()


def shape_mask(mask, target_w, max_h):
    """Resize / stretch / rotate / thicken a cropped mask to fit on the page."""
    h, w = mask.shape
    stretch = random.uniform(0.75, 1.25)
    nw, nh = target_w, max(4, int(h * target_w / w * stretch))
    if nh > max_h:
        nw, nh = max(8, int(nw * max_h / nh)), max_h
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = rotate(mask, random.uniform(-12, 12))
    r = random.random()
    if r < 0.2:
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8))
    elif r < 0.35:
        thin = cv2.erode(mask, np.ones((2, 2), np.uint8))
        if np.count_nonzero(thin > 30) > 0.3 * np.count_nonzero(mask > 30):
            mask = thin
    return crop(mask)


# ---------------------------------------------------------------- distractors

def arc_text(canvas, text, cx, cy, radius, a0, a1, top, scale, thick, font):
    """Draw text along a circular arc (angles in degrees, image coordinates)."""
    if len(text) < 2:
        return
    for i, ch in enumerate(text):
        a = np.deg2rad(a0 + (a1 - a0) * i / (len(text) - 1))
        (tw, th), _ = cv2.getTextSize(ch, font, scale, thick)
        s = max(tw, th) * 2 + 4
        patch = np.zeros((s, s), np.uint8)
        cv2.putText(patch, ch, ((s - tw) // 2, (s + th) // 2), font, scale, 255, thick, cv2.LINE_AA)
        ang = 270 - np.rad2deg(a) if top else 90 - np.rad2deg(a)
        patch = cv2.warpAffine(patch, cv2.getRotationMatrix2D((s / 2, s / 2), ang, 1.0), (s, s))
        x, y = int(cx + radius * np.cos(a) - s / 2), int(cy + radius * np.sin(a) - s / 2)
        if 0 <= x and 0 <= y and x + s <= canvas.shape[1] and y + s <= canvas.shape[0]:
            canvas[y:y + s, x:x + s] = np.maximum(canvas[y:y + s, x:x + s], patch)


def make_stamp():
    font = random.choice(HERSHEY)
    word = random.choice(STAMP_WORDS)
    small = random.choice(SMALL_WORDS)
    date = random.choice([f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-{random.randint(1995, 2026)}",
                          f"{random.randint(1, 28):02d} {random.choice(MONTHS)} {random.randint(1995, 2026)}"])
    th = random.randint(2, 4)
    kind = random.choice(["rect", "rect", "circle", "circle", "date"])

    if kind == "circle":
        R = random.randint(110, 160)
        m = np.zeros((2 * R + 20, 2 * R + 20), np.uint8)
        c = R + 10
        cv2.circle(m, (c, c), R, 255, th + 1, cv2.LINE_AA)
        if random.random() < 0.8:
            cv2.circle(m, (c, c), int(R * 0.68), 255, th, cv2.LINE_AA)
        arc_text(m, small, c, c, R * 0.84, 215, 325, True, 0.8, 2, font)
        if random.random() < 0.6:
            arc_text(m, random.choice(SMALL_WORDS), c, c, R * 0.84, 145, 35, False, 0.7, 2, font)
        text = random.choice([word, date]) if random.random() < 0.3 else word
        scale = 1.0
        (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
        scale *= min(1.2, 1.25 * R / tw)
        (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
        if random.random() < 0.5:   # band across the middle
            cv2.rectangle(m, (c - int(R * 0.95), c - tht), (c + int(R * 0.95), c + tht), 0, -1)
            cv2.rectangle(m, (c - int(R * 0.95), c - tht), (c + int(R * 0.95), c + tht), 255, th)
        cv2.putText(m, text, (c - tw // 2, c + tht // 2), font, scale, 255, th, cv2.LINE_AA)
    else:
        lines = [(word, random.uniform(1.4, 2.2))]
        if kind == "date":
            lines = [(small, 0.9), (date, random.uniform(1.2, 1.6))]
        elif random.random() < 0.6:
            lines.append((random.choice([small, date]), 0.9))
        sizes = [cv2.getTextSize(t, font, s, th)[0] for t, s in lines]
        pad = random.randint(14, 26)
        w = max(s[0] for s in sizes) + 2 * pad
        h = sum(s[1] for s in sizes) + pad * (len(lines) + 1)
        m = np.zeros((h + 20, w + 20), np.uint8)
        cv2.rectangle(m, (10, 10), (w + 10, h + 10), 255, th + 1)
        if random.random() < 0.4:
            cv2.rectangle(m, (16, 16), (w + 4, h + 4), 255, max(1, th - 1))
        y = 10 + pad
        for (t, s), (tw, tht) in zip(lines, sizes):
            y += tht
            cv2.putText(m, t, (10 + (w - tw) // 2, y), font, s, 255, th, cv2.LINE_AA)
            y += pad
    m = wear(m, random.uniform(0.6, 0.95))
    return crop(rotate(m, random.uniform(-35, 35)))


def blend(page, mask, x, y, color, opacity):
    """Multiply-blend ink onto the page so existing print stays visible."""
    h, w = mask.shape
    roi = page[y:y + h, x:x + w].astype(np.float32)
    a = (mask.astype(np.float32) / 255 * opacity)[..., None]
    col = np.array(color, np.float32) / 255
    page[y:y + h, x:x + w] = np.clip(roi * (1 - a * (1 - col)), 0, 255).astype(np.uint8)


def sign_line(page, x, y, length):
    """Printed 'Signature: ______' line (unlabelled)."""
    H, W = page.shape[:2]
    x2 = min(W - 2, x + length)
    col = (random.randint(0, 60),) * 3
    if random.random() < 0.8:
        cv2.line(page, (x, y), (x2, y), col, random.choice([1, 1, 2]))
    else:
        for xx in range(x, x2, 8):
            cv2.line(page, (xx, y), (min(x2, xx + 4), y), col, 1)
    label = random.choice(SIGN_LABELS)
    if label and y + 22 < H:
        cv2.putText(page, label, (x, y + 18), random.choice(HERSHEY[:2]),
                    random.uniform(0.4, 0.6), col, 1, cv2.LINE_AA)


def tick(page):
    H, W = page.shape[:2]
    s = random.randint(12, 30)
    x, y = random.randint(0, W - s - 1), random.randint(0, H - s - 1)
    pts = np.array([[x, y + s * 0.55], [x + s * 0.4, y + s], [x + s, y]] if random.random() < 0.6
                   else [[x, y], [x + s, y + s]], np.int32)
    cv2.polylines(page, [pts], False, random.choice(INK_COLORS), 2, cv2.LINE_AA)


# ---------------------------------------------------------------- one image

def overlaps(b, boxes, margin=6):
    return any(b[0] < o[2] + margin and o[0] < b[2] + margin and
               b[1] < o[3] + margin and o[1] < b[3] + margin for o in boxes)


def make_one(task):
    i, page_path, subset, out, seed = task
    random.seed(seed + i)
    np.random.seed((seed + i) % 2 ** 32)
    try:
        page = cv2.imread(page_path)
        if page is None:
            return 0
        if random.random() < 0.5:
            f = random.uniform(0.8, 1.5)
            page = cv2.resize(page, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        H, W = page.shape[:2]

        # empty signature lines (negatives)
        for _ in range(random.choice([0, 0, 1, 2])):
            sign_line(page, random.randint(0, W // 2), random.randint(H // 3, H - 30),
                      int(W * random.uniform(0.2, 0.4)))

        boxes = []
        n_sig = 0 if random.random() < 0.25 else random.choice([1, 1, 1, 2, 2, 3])
        for _ in range(n_sig):
            for _attempt in range(10):
                mask = make_signature(subset)
                if mask is None:
                    continue
                mask = shape_mask(mask, int(W * random.uniform(0.08, 0.30)), int(H * 0.12))
                if mask is None:
                    continue
                h, w = mask.shape
                if w >= W - 4 or h >= H - 4:
                    continue
                on_line = random.random() < 0.6
                if on_line:
                    x = random.randint(int(W * 0.03), max(int(W * 0.03), W - w - 2))
                    line_y = random.randint(int(H * 0.3), H - 40)
                    y = min(max(0, line_y - h + int(h * random.uniform(0, 0.3))), H - h - 1)
                else:
                    x, y = random.randint(0, W - w - 1), random.randint(0, H - h - 1)
                box = (x, y, x + w, y + h)
                if overlaps(box, boxes):
                    continue
                if on_line:
                    lx = max(0, x - int(w * random.uniform(0, 0.2)))
                    sign_line(page, lx, line_y, int(w * random.uniform(1.1, 1.6)))
                faded = random.random() < 0.15
                if faded:
                    mask = wear(mask, random.uniform(0.7, 0.95))
                opacity = random.uniform(0.25, 0.55) if faded else random.uniform(0.75, 1.0)
                blend(page, mask, x, y, random.choice(INK_COLORS), opacity)
                boxes.append(box)
                break

        # stamps (negatives), sometimes right on top of a signature
        for _ in range(random.choice([0, 0, 1, 1, 1, 2, 3])):
            st = make_stamp()
            if st is None:
                continue
            tw = int(W * random.uniform(0.12, 0.30))
            st = cv2.resize(st, (tw, max(4, int(st.shape[0] * tw / st.shape[1]))), interpolation=cv2.INTER_AREA)
            sh, sw = st.shape
            if sh >= H or sw >= W:
                continue
            if boxes and random.random() < 0.3:
                b = random.choice(boxes)
                cx, cy = (b[0] + b[2]) // 2 + random.randint(-sw // 2, sw // 2), (b[1] + b[3]) // 2
            else:
                cx, cy = random.randint(sw // 2, W - sw // 2), random.randint(sh // 2, H - sh // 2)
            x, y = min(max(0, cx - sw // 2), W - sw), min(max(0, cy - sh // 2), H - sh)
            blend(page, st, x, y, random.choice(STAMP_COLORS), random.uniform(0.5, 0.95))

        for _ in range(random.choice([0, 0, 0, 1, 2])):
            tick(page)

        # scan degradation
        if random.random() < 0.2:
            page = cv2.cvtColor(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        page = np.clip(page.astype(np.float32) * random.uniform(0.85, 1.1) + random.uniform(-15, 15), 0, 255)
        r = random.random()
        if r < 0.35:
            page = cv2.GaussianBlur(page, (3, 3), 0)
        elif r < 0.5:
            f = random.uniform(0.5, 0.8)
            page = cv2.resize(cv2.resize(page, None, fx=f, fy=f), (W, H))
        page = np.clip(page + np.random.normal(0, random.uniform(0, 6), page.shape), 0, 255).astype(np.uint8)

        name = f"img_{i:05d}"
        cv2.imwrite(f"{out}/images/{subset}/{name}.jpg", page, [cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 95)])
        with open(f"{out}/labels/{subset}/{name}.txt", "w") as f:
            f.write("\n".join(f"0 {(a + c) / 2 / W:.6f} {(b + d) / 2 / H:.6f} {(c - a) / W:.6f} {(d - b) / H:.6f}"
                              for a, b, c, d in boxes))
        return 1
    except Exception as e:
        print(f"skip {page_path}: {e}", flush=True)
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="pages")
    ap.add_argument("--signatures", default="signatures")
    ap.add_argument("--fonts", default="fonts")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-every", type=int, default=10, help="every Nth page goes to validation")
    ap.add_argument("--workers", type=int, default=cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pages = sorted(glob.glob(f"{args.pages}/*.jpg") + glob.glob(f"{args.pages}/*.png"))
    init(args.signatures, args.fonts)
    print(f"Pages: {len(pages)} | CEDAR train/val: {len(SIGS['train'])}/{len(SIGS['val'])} "
          f"| fonts train/val: {len(FONTS['train'])}/{len(FONTS['val'])}")
    assert pages and SIGS["train"], "pages/ or signatures/ is empty"

    for s in ["train", "val"]:
        os.makedirs(f"{args.out}/images/{s}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{s}", exist_ok=True)

    tasks = [(i, p, "val" if i % args.val_every == 0 else "train", args.out, args.seed)
             for i, p in enumerate(pages)]
    done = 0
    with Pool(args.workers, initializer=init, initargs=(args.signatures, args.fonts)) as pool:
        for k, ok in enumerate(pool.imap_unordered(make_one, tasks, chunksize=32), 1):
            done += ok
            if k % 2000 == 0:
                print(k, "/", len(tasks), flush=True)

    with open(f"{args.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: signature\n")
    print(f"Done: {done} images -> {args.out}")


if __name__ == "__main__":
    main()
