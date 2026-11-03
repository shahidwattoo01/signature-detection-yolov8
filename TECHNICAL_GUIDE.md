# Signature Detection – Technical Guide

A YOLOv8 object-detection model that finds handwritten signatures in scanned documents and PDFs, and ignores look-alikes such as rubber stamps, printed signature lines and tick marks. It was trained on **50,000 synthetic, automatically labelled document images** and scores **0.941 mAP50** on held-out validation data.

---

## 1. Results

### Validation (5,000 images, 6,300 signatures, writers and fonts never seen in training)

| Metric | Value | Meaning |
|---|---|---|
| Precision | **0.973** | 97% of the boxes the model draws are real signatures |
| Recall | **0.883** | It finds 88% of all signatures, including very faded and tiny ones |
| mAP50 | **0.941** | Overall detection quality, counting a box as correct at IoU ≥ 0.5 |
| mAP50-95 | **0.789** | How tight the boxes are, averaged over IoU 0.5–0.95 |
| Speed (Tesla T4) | 2.5 ms / image | Inference only; about 0.1–0.3 s per page on a normal CPU |
| Model size | 3.13 M parameters, 6.5 MB | YOLOv8n (nano), 8.6 GFLOPs |

### Test PDF (`test_signature_document.pdf`, 10 pages, 12 signatures, 9 stamps)

| | First model (10k images, no stamps in training) | Final model (`best_50k.pt`) |
|---|---|---|
| Signatures found | 4 of 12 | **12 of 12** |
| Stamps wrongly boxed | 6 | **0** |
| Faded signature (page 8) | missed | found (0.89) |
| Short "VR" and zigzag signatures (pages 3, 4) | missed | found (0.62, 0.91) |

---

## 2. Pipeline at a glance

1. **Download backgrounds:** 50,000 real scanned document pages from RVL-CDIP (`get_pages.py`). Classes that often contain real handwriting are skipped.
2. **Collect signature sources:** CEDAR signature scans, about 18 handwriting fonts (`get_fonts.py`), and signatures generated from mathematical curves.
3. **Generate the dataset:** `make_dataset.py` blends 0–3 signatures onto each page, adds unlabelled stamps, lines and ticks, simulates scanning, and writes a YOLO label file per image. The output is 45,000 train and 5,000 val images.
4. **Train:** YOLOv8n is fine-tuned for 40 epochs on a free Colab T4 GPU (`train_colab.ipynb`). Checkpoints go to Google Drive, so training survives disconnects.
5. **Evaluate:** precision, recall and mAP on the held-out validation set, plus a visual check on a real PDF.
6. **Detect:** `detect.py` (command line) or `app.py` (localhost web app) renders PDF pages, runs the model, and outputs annotated pages plus cropped signatures.

```
RVL-CDIP pages ───┐
CEDAR signatures ─┤
Handwriting fonts ┼─► make_dataset.py ─► 50k images + YOLO labels ─► YOLOv8n training (Colab T4)
Math scribbles ───┤                                                         │
Fake stamps ──────┘                                                     best_50k.pt
                                                                            │
PDF / image ─► PyMuPDF render ─► YOLO forward pass ─► confidence filter + NMS ─► boxes + crops
```

---

## 3. Project files

| File / folder | Purpose |
|---|---|
| `app.py` | Gradio web app on http://127.0.0.1:7860: upload a document, see boxes, crops and a table, download a ZIP |
| `detect.py` | Command-line detection on a PDF or image; writes to `output/` |
| `best_50k.pt` | **Final trained model** (6.5 MB) |
| `best.pt` | First model (10k images), kept only for before/after comparison |
| `make_dataset.py` | Synthetic dataset generator (all the computer-vision augmentation) |
| `get_pages.py` | Downloads background pages from RVL-CDIP |
| `get_fonts.py` | Downloads the handwriting fonts |
| `train_colab.ipynb` | End-to-end Colab notebook: build data on CPU, train on GPU, test |
| `check_labels.py` | Draws the label boxes on a random generated image, to check labels |
| `signature.yaml` | YOLO dataset config (one class: `signature`) |
| `requirements.txt` | Python packages |
| `test_signature_document.pdf` | 10-page test document |
| `signatures/` and `signatures.zip` | CEDAR signature scans (the same data twice; one is enough). Only needed to rebuild the dataset |
| Google Drive: `signature_project/runs/sig50k/` | Training graphs and checkpoints: `results.png`, `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, `val_batch*_pred.jpg` |
| Google Drive: `signature_dataset_50k.zip` | The generated 50k dataset (~3–4 GB). Only needed to retrain |

---

## 4. Data sources

| Source | What it is | Role |
|---|---|---|
| **RVL-CDIP** (Hugging Face `chainyo/rvl-cdip`) | 400,000 grayscale scanned business documents in 16 classes, max 1000 px | Real page backgrounds |
| **CEDAR** | 55 writers × 24 genuine + 24 forged signatures = 2,640 scans | Real handwriting. Forgeries are still signatures, so they are useful for detection |
| **Google Fonts** (Great Vibes, Allura, Sacramento, Homemade Apple…) | ~18 handwriting fonts, open licence | Font-written names in styles CEDAR lacks |
| **Procedural strokes** | Curves generated in code | Short, zigzag and looped signatures |

**Why synthetic data:** labelling tens of thousands of real signed documents by hand is slow, and real signed documents are private. When code pastes a signature onto a page, it knows the exact box, so every label is free and pixel-accurate. The risk is the *domain gap* (fake data not looking real), which is addressed with varied styles, distractors and scan effects.

**Skipped RVL-CDIP classes:** letter, form, email, handwritten, questionnaire, resume, memo. These often contain real signatures or handwriting. Left in, they would appear in training as handwriting *without* a label, teaching the model that signatures are background (*label noise*).

---

## 5. Synthetic data generation (`make_dataset.py`)

Basics used throughout:
- An image is a NumPy array: grayscale `H × W` (0 = black, 255 = white), or colour `H × W × 3` in OpenCV's **BGR** order.
- A **mask** is a grayscale image where 255 = ink and 0 = empty. Every signature and stamp is built as a mask, then coloured and blended onto the page.

### 5.1 Setup
- **Writer-based split:** CEDAR file `original_12_3.png` = writer 12. Writers 10, 20, 30, 40 and 50 go only to validation, and every 6th font is validation-only. This prevents *leakage*: validation tests generalisation to new handwriting, not memory of seen writers.
- **Page split:** every 10th page goes to validation, giving 45,000 train and 5,000 val.
- **Parallel:** `multiprocessing.Pool` with one worker per CPU core.
- **Reproducible:** image *i* is generated with `random.seed(42 + i)`.

### 5.2 Signature source 1: CEDAR scan → mask
1. Read in grayscale.
2. **Otsu thresholding** (`THRESH_BINARY_INV + THRESH_OTSU`). Otsu picks the brightness cut-off automatically by finding the threshold that best separates the two histogram peaks (ink vs paper). INV makes ink = 255.
3. **Morphological opening** (erode, then dilate, 2×2 kernel) removes dust specks but keeps strokes.
4. **Gaussian blur 3×3** softens jagged threshold edges (anti-aliasing).
5. **Tight crop** to the min/max coordinates of ink pixels. This crop *is* the future label box.

### 5.3 Signature source 2: handwriting font
1. Random name formats: "Sara Khan", "S. Khan", "SK", "Khan" and so on.
2. Render with Pillow at 60–120 px, white text on black.
3. In 35% of cases add an underline flourish: a 3-point curve smoothed by Chaikin's algorithm, drawn anti-aliased with `cv2.polylines(..., LINE_AA)`.

### 5.4 Signature source 3: procedural strokes
- **Zigzag:** x steps right randomly; y alternates ±amplitude plus noise.
- **Flowing:** y is a *random walk* (each point = previous + random step) plus a sine wave, then **Chaikin corner-cutting** applied 3 times. Each segment is replaced by points at 25% and 75% along it, which makes a smooth curve.
- **Looped:** a **trochoid**, `x = a·t − r·sin(t)`, `y = −r·cos(t)`. When r > a the curve forms cursive-like loops.
- Stroke width is 2–5 px, with an optional underline or crossing stroke.

### 5.5 Shaping each signature
- **Width:** 8–30% of the page width; aspect stretch 0.75–1.25; height capped at 12% of the page.
- **Resize** with `INTER_AREA`, the best method for shrinking because thin strokes survive.
- **Rotate** ±12° with an affine matrix (`cv2.getRotationMatrix2D`). The canvas is expanded to `w·|cos θ| + h·|sin θ|` by `w·|sin θ| + h·|cos θ|` so corners aren't cut off.
- **Pen thickness:** 20% dilate (felt pen); 15% erode (fine pen), kept only if at least 30% of the ink survives.

### 5.6 Placing signatures on the page
- 50% of pages are first rescaled 0.8–1.5× for resolution variety.
- **Count:** 25% of pages get none (*negative examples*); the rest get 1–3.
- **60% on a printed signature line** with a label such as "Signature:" or "Witness:", the signature dipping slightly below the line. **40% anywhere**, including over text.
- **No overlap** between signatures (box check with a 6 px margin, up to 10 retries).
- **Faded ink (15%):** a smooth random *noise field* (small random grid upscaled with cubic interpolation) knocks out blotches, and opacity drops to 0.25–0.55.
- **Ink colours:** black, navy, blue, red, green, purple.
- **Multiply blending:** `new = old × (1 − a × (1 − ink/255))`, where `a = mask/255 × opacity`. On white paper this gives the ink colour; over black text it stays black, so printed text shows through the ink as with real pen strokes, instead of a pasted-sticker look.

### 5.7 Distractors (hard negatives, never labelled)
The model is penalised every time it boxes one of these, which taught it to ignore stamps. This was the main fix over the first model.
- **Rubber stamps (0–3 per page):**
  - Rectangular, double-bordered, date and circular styles, with words like RECEIVED, PAID, APPROVED, VERIFIED and CONFIDENTIAL, plus office names and dates.
  - Circular stamps have **text along an arc**: each letter is placed at `centre + r·(cos a, sin a)` and rotated by `270° − a`, so it faces outward.
  - Worn with the noise field, rotated ±35°, in stamp colours at opacity 0.5–0.95.
  - 30% are placed **on top of a signature**, so the model learns to find the signature under the stamp without boxing the stamp.
- **Empty signature lines (0–2):** a line alone is not a signature.
- **Tick and X marks (0–2):** small pen marks are not signatures.

### 5.8 Scan simulation (whole page)
Grayscale (20%), brightness/contrast jitter, Gaussian blur (35%) or down-and-up resampling (15%), Gaussian noise (σ 0–6), and JPEG compression at quality 55–95.

### 5.9 YOLO label format
One `.txt` per image, one line per signature: `class x_center y_center width height`, all normalised to 0–1. An empty file means a background image.

Example: a page 800 × 1000 px with a box from x 100–300 and y 700–760 gives `0 0.250000 0.730000 0.250000 0.060000`.

```
dataset/
  images/train (45,000 .jpg)   labels/train (45,000 .txt)
  images/val   (5,000 .jpg)    labels/val   (5,000 .txt)
  data.yaml
```

---

## 6. Model: how YOLOv8 object detection works

### Key concepts
| Term | Meaning |
|---|---|
| Object detection | Find *where* each object is (box + class + confidence), unlike classification (one label per image) or segmentation (per-pixel mask) |
| Bounding box | Rectangle `(x1, y1, x2, y2)` |
| Confidence | Model's certainty (0–1) that a box is a signature |
| IoU | Overlap area ÷ union area of two boxes; 1 = identical |
| NMS | Non-Maximum Suppression: keeps the highest-confidence box and removes overlapping duplicates (IoU > 0.7) |

**YOLO = You Only Look Once:** a single forward pass of a convolutional neural network outputs all boxes at once. Older two-stage detectors first proposed regions and then classified each one, which was slower.

### Architecture (YOLOv8n, 3.13 M parameters)
1. **Backbone (feature extractor):**
   - Convolution layers with **C2f** blocks (cross-stage partial connections) learn edges, then strokes, then shapes.
   - Resolution halves per stage: 640 → 320 → 160 → 80 → 40 → 20.
   - **SPPF** (spatial pyramid pooling) adds wide context.
2. **Neck (FPN + PAN):** a Feature Pyramid Network passes high-level meaning down to high-resolution maps; a Path Aggregation Network passes precise location back up.
3. **Head (decoupled, anchor-free):** predictions on three grids. Every cell predicts a class score (sigmoid) and the distances from the cell centre to the four box edges. **DFL (Distribution Focal Loss)** predicts each distance as a probability distribution, which gives sharper edges.

| Grid | Stride | Cells | Detects |
|---|---|---|---|
| 80 × 80 | 8 px | 6,400 | small signatures and initials |
| 40 × 40 | 16 px | 1,600 | medium |
| 20 × 20 | 32 px | 400 | large |

That makes 8,400 candidate predictions per image; the confidence threshold and NMS reduce them to the real signatures.

---

## 7. Training (Google Colab)

### Two-phase notebook
1. **CPU runtime** (saves free GPU time): download 50k pages, download fonts, unzip CEDAR, generate the dataset, then zip it to Google Drive (`signature_dataset_50k.zip`, stored uncompressed because JPEGs are already compressed). This took about 3–4 hours.
2. **T4 GPU runtime:** restore the dataset from Drive (minutes), then train. Checkpoints are written to Drive every epoch (`last.pt`, `best.pt`). After a disconnect, *Run all* resumes from `last.pt` with the same optimizer state. The run spanned several sessions.

### Settings
| Setting | Value | Why |
|---|---|---|
| Starting weights | first model `best.pt` | **Transfer learning / fine-tuning**: features already learned, faster convergence |
| epochs | 40 | ~13–14 min each on T4 (1,407 batches) |
| imgsz | 640 | network input; pages are letterboxed (aspect kept, gray padding) |
| batch | 32 | fits the T4's 15 GB (≈4–5 GB used) |
| optimizer | auto → MuSGD, lr 0.01, momentum 0.937 | chosen by Ultralytics for long runs; warmup in the first 3 epochs, then LR decay |
| patience | 10 | early stopping if validation stops improving |
| close_mosaic | 5 | last 5 epochs without mosaic, to finish on realistic full pages |
| fliplr | 0.0 | mirrored stamps and text don't exist in real documents |
| AMP | on | mixed precision, ~2× faster on T4 |

### Built-in augmentation (on top of the synthetic data)
**Mosaic** (4 images stitched into 1), HSV colour jitter, random scale (±50%) and translation.

### One training step
1. Load a batch of images and labels.
2. Augment and letterbox to 640.
3. Forward pass → 8,400 predictions per image.
4. **Task-Aligned Assigner:** choose the predictions responsible for each true box, scored by class confidence × IoU.
5. **Loss:**
   - **CIoU** box loss (overlap, centre distance, aspect ratio)
   - **DFL** loss (edge distributions)
   - **BCE** classification loss: positives → 1, everything else → 0, which is what penalises boxes on stamps
6. **Backpropagation** computes the gradient for every weight; the optimizer updates the weights.

### Metrics
- **Precision** = correct boxes ÷ all boxes (false alarms lower it).
- **Recall** = found signatures ÷ all signatures (misses lower it).
- **mAP50 / mAP50-95** = area under the precision–recall curve, at IoU 0.5 or averaged over 0.5–0.95.
- `best.pt` is saved at the best validation fitness.

---

## 8. Inference

### `detect.py`
1. Parse arguments: source, `--model`, `--conf` (0.4), `--out`, `--dpi` (100).
2. **PDF → images with PyMuPDF:** each page is rasterised at 100 DPI (≈850 × 1100 px, similar to the training page scale). The pixel buffer becomes a NumPy array and is converted RGB → BGR.
3. **Model call:** letterbox to 640 → normalise to 0–1 → forward pass → decode boxes → drop confidence below 0.4 → NMS → scale boxes back to page pixels.
4. Save the annotated page (`page_N.jpg`). Clip boxes to the page edges and save each crop (`page_N_signature_M.png`).

### `app.py` (localhost web app, Gradio)
- Upload a PDF, PNG or JPG; adjust the confidence slider; view the annotated pages, a gallery of cropped signatures, and a table (page, confidence, coordinates); download all crops as a ZIP.
- It reuses `load_pages()` from `detect.py` and runs fully offline on CPU.
- It can be deployed unchanged to Hugging Face Spaces.

**Confidence threshold trade-off:** lower (0.25) catches fainter signatures but risks false alarms; higher (0.6) gives fewer false alarms but more misses.

---

## 9. How to run it

```bash
pip install -r requirements.txt
```

**Web app:**
```bash
python app.py
```
Then open http://127.0.0.1:7860. Stop with Ctrl+C.

**Command line:**
```bash
python detect.py test_signature_document.pdf --model best_50k.pt
```

**Rebuild the dataset and retrain:**
- Run `train_colab.ipynb` (upload `signatures.zip`, `best.pt`, `test_signature_document.pdf`, `get_pages.py`, `make_dataset.py` and `detect.py` to `MyDrive/signature_project`), or
- locally:
```bash
python get_pages.py --count 50000 --out pages
python get_fonts.py
python make_dataset.py --pages pages --signatures signatures --fonts fonts --out dataset
yolo detect train data=dataset/data.yaml model=best.pt epochs=40 imgsz=640
```

---

## 10. Limitations, next steps, likely questions

### Limitations
- **Synthetic-to-real gap:** the test PDF is a sanity check, not a benchmark. A fair real-world score needs 100+ real documents labelled by hand (e.g. in CVAT or Roboflow).
- **Detection, not verification:** the model finds signatures; it does not tell who signed or whether a signature is forged.
- **Possible false positives:** dense handwritten notes or handwritten form fields were not added as negatives.
- **Background pages:** some kept RVL-CDIP classes (invoices, budgets) may still contain unlabelled real signatures.
- **`get_pages.py` version:** the shared `get_pages.py` is a faster rewrite (downloads whole data shards instead of streaming). The 50k dataset was built with the earlier streaming version, and the rewrite has not been run end to end yet.

### Next steps
1. Build a small real labelled test set and report real-world precision and recall.
2. Add handwritten-text negatives.
3. Add a **verification** stage: a Siamese network on the crops, trained on CEDAR's genuine/forged pairs.
4. Export to ONNX for faster CPU inference, and deploy the Gradio app on Hugging Face Spaces.

### Likely questions in a presentation
| Question | Short answer |
|---|---|
| Why synthetic data? | Real labelled signed documents are scarce and private; synthetic labels are free and exact |
| How do you stop it detecting stamps? | Generated stamps are added as unlabelled hard negatives, so boxing them increases the loss |
| How do you know it generalises? | Validation uses writers and fonts never seen in training |
| Why YOLOv8n? | Real-time, 6.5 MB, runs on CPU; a single class doesn't need a big model |
| What do the metrics mean? | 97% of detections are correct, 88% of signatures are found, 0.94 mAP50 |
| How long did training take? | ~9 h on a free Colab T4 for 40 epochs over 45k images |
| Can it run offline? | Yes. The web app and command line run locally on CPU |
