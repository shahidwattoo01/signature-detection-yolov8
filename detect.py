"""Find signatures in a PDF or image.

Usage: python detect.py document.pdf [--model best.pt] [--conf 0.4] [--out output]
Writes annotated pages and cropped signatures to the output folder.
"""
import argparse, glob, os
import cv2
import fitz  # PyMuPDF
import numpy as np
from ultralytics import YOLO


def load_pages(path, dpi):
    """Return a list of page images (BGR) from a PDF or an image file."""
    if path.lower().endswith(".pdf"):
        pages = []
        for p in fitz.open(path):
            pix = p.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            pages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR))
        return pages
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read {path}")
    return [img]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="PDF or image file")
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default="output")
    ap.add_argument("--dpi", type=int, default=100, help="PDF render DPI (100 matches the training page scale)")
    args = ap.parse_args()

    model = YOLO(args.model)
    os.makedirs(args.out, exist_ok=True)
    for old in glob.glob(f"{args.out}/page_*"):
        os.remove(old)

    total = 0
    for i, page in enumerate(load_pages(args.source, args.dpi)):
        result = model(page, conf=args.conf, imgsz=640, verbose=False)[0]
        cv2.imwrite(f"{args.out}/page_{i+1}.jpg", result.plot())

        H, W = page.shape[:2]
        for j, box in enumerate(result.boxes.xyxy.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = np.clip(box, 0, [W, H, W, H])
            cv2.imwrite(f"{args.out}/page_{i+1}_signature_{j+1}.png", page[y1:y2, x1:x2])
            conf = float(result.boxes.conf[j])
            print(f"Page {i+1}: signature at ({x1},{y1},{x2},{y2}), confidence {conf:.2f}")
            total += 1
    print(f"Done. {total} signature(s) found. See the '{args.out}' folder.")


if __name__ == "__main__":
    main()
