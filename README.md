---
title: Signature Detection
emoji: ✍️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
---

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
Real documents with labelled signatures are scarce, so the training data is synthetic: the pipeline renders random signatures from a font pool, stamps, printed lines and tick marks onto realistic document backgrounds with scan effects. A held-out writer pool and font pool test generalisation. Full details in [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).
