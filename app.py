"""Web demo: upload a PDF or image, see detected signatures.

Usage: python app.py   -> open http://127.0.0.1:7860
"""
import os, tempfile, zipfile
import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO
from detect import load_pages

MODEL_PATH = os.environ.get("MODEL", "best_50k.pt" if os.path.exists("best_50k.pt") else "best.pt")
model = YOLO(MODEL_PATH)


def run(file, conf):
    if file is None:
        raise gr.Error("Upload a PDF or image first.")
    pages = load_pages(file, dpi=100)

    annotated, crops, rows = [], [], []
    zip_path = os.path.join(tempfile.mkdtemp(), "signatures.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i, page in enumerate(pages, 1):
            result = model(page, conf=conf, imgsz=640, verbose=False)[0]
            annotated.append((cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), f"Page {i}"))

            H, W = page.shape[:2]
            for j, (box, c) in enumerate(zip(result.boxes.xyxy.cpu().numpy().astype(int),
                                             result.boxes.conf.cpu().numpy()), 1):
                x1, y1, x2, y2 = np.clip(box, 0, [W, H, W, H])
                crop = page[y1:y2, x1:x2]
                name = f"page_{i}_signature_{j}.png"
                zf.writestr(name, cv2.imencode(".png", crop)[1].tobytes())
                crops.append((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), f"Page {i} #{j} ({c:.2f})"))
                rows.append([i, j, round(float(c), 2), int(x1), int(y1), int(x2), int(y2)])

    summary = f"**{len(rows)} signature(s)** found on {len(pages)} page(s) using `{MODEL_PATH}`."
    return summary, annotated, crops, rows, zip_path if rows else None


with gr.Blocks(title="Signature Detection") as demo:
    gr.Markdown("# Signature Detection\nUpload a scanned document (PDF, PNG or JPG). "
                "The YOLOv8 model finds handwritten signatures and ignores stamps and printed lines.")
    with gr.Row():
        with gr.Column(scale=1):
            file = gr.File(label="Document", file_types=[".pdf", ".png", ".jpg", ".jpeg"], type="filepath")
            conf = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence threshold")
            btn = gr.Button("Detect signatures", variant="primary")
            summary = gr.Markdown()
            zip_out = gr.File(label="Download cropped signatures (ZIP)")
        with gr.Column(scale=2):
            pages_out = gr.Gallery(label="Pages", columns=2, height=600)
    crops_out = gr.Gallery(label="Cropped signatures", columns=6, height=220)
    table = gr.Dataframe(headers=["page", "#", "confidence", "x1", "y1", "x2", "y2"], label="Detections")

    btn.click(run, [file, conf], [summary, pages_out, crops_out, table, zip_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
