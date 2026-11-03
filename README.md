# Signature Detection (YOLOv8)

Detects handwritten signatures in scanned documents and PDFs, and ignores look-alikes such as rubber stamps, printed signature lines and tick marks. Full details are in [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).

## Results (`best_50k.pt`, 5,000 held-out validation images)
| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.973 | 0.883 | 0.941 | 0.789 |

On the 10-page test PDF it finds all 12 signatures and boxes none of the 9 stamps. The first model found 4 signatures and boxed 6 stamps.

## Quick start
```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:7860 and upload a PDF or image.

## How it works
Real documents with labelled signatures are scarce, so the training data is synthetic:

1. **Backgrounds:** 50,000 document pages from [RVL-CDIP](https://huggingface.co/datasets/chainyo/rvl-cdip). Classes that often contain real handwriting (letters, forms, memos and so on) are skipped, so they can't become unlabelled signatures.
2. **Signatures** (labelled), 0–3 per page, from three sources:
   - CEDAR signature scans (55 writers)
   - names rendered in about 18 handwriting fonts
   - random pen strokes (flowing, zigzag and looped)

   Each signature gets a random size, rotation, stroke width, ink colour and fading. It's usually placed on a printed "Signature: ____" line.
3. **Distractors** (not labelled):
   - generated rubber stamps (rectangular, circular and date stamps, worn and rotated, sometimes over a signature)
   - empty signature lines
   - tick marks
4. **Scan effects:** blur, resampling, noise, contrast changes and JPEG compression.
5. **Validation split:** 45k train / 5k val. Validation uses CEDAR writers and fonts that are never seen in training.

## Files
| File | Purpose |
|---|---|
| `get_pages.py` | Download background pages from RVL-CDIP |
| `make_dataset.py` | Generate the YOLO dataset (parallel) |
| `train_colab.ipynb` | End-to-end pipeline on a Colab T4: build the data, train, test |
| `detect.py` | Run the model on a PDF or image, save annotated pages and crops |
| `app.py` | Localhost web app (Gradio) |
| `get_fonts.py` | Download the handwriting fonts |
| `check_labels.py` | Draw the label boxes on a random generated image |
| `signature.yaml` | Dataset config |

## Signature data
The CEDAR signature scans are not included in this repository (third-party data). Download the CEDAR signature dataset (University at Buffalo) and put its `full_org` and `full_forg` folders under `signatures/`. For Colab, zip that folder as `signatures.zip`. Detection only needs `best_50k.pt`.

## Train (Google Colab)
1. Upload `signatures.zip`, `best.pt`, `test_signature_document.pdf`, `get_pages.py`, `make_dataset.py` and `detect.py` to `MyDrive/signature_project`.
2. Open `train_colab.ipynb`, choose a T4 GPU and run all cells. The dataset and checkpoints are saved to Drive, so after a disconnect you just run all again.

## Train locally
```bash
pip install -r requirements.txt
python get_pages.py --count 50000 --out pages
python make_dataset.py --pages pages --signatures signatures --fonts fonts --out dataset
yolo detect train data=dataset/data.yaml model=best.pt epochs=40 imgsz=640
```

## Detect
```bash
python detect.py test_signature_document.pdf --model best_50k.pt
```
Results are written to `output/`: `page_N.jpg` (annotated page) and `page_N_signature_M.png` (cropped signature).
