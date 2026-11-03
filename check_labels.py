import cv2, glob, random, sys

# Usage: python check_labels.py [dataset_dir]  -> draws label boxes on a random image into check.jpg
root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
img_path = random.choice(glob.glob(f"{root}/images/*/*.jpg"))
lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

img = cv2.imread(img_path)
H, W = img.shape[:2]

for line in open(lbl_path).read().strip().splitlines():
    if not line.strip():
        continue
    _, xc, yc, w, h = map(float, line.split())
    x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
    x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imwrite("check.jpg", img)
print(img_path)
