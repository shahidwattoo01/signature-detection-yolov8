"""Download document pages (backgrounds for the synthetic dataset) from RVL-CDIP.

Downloads whole parquet shards (~490 MB, ~2,700 pages each) one at a time, keeps the
pages we want and deletes the shard. Much faster than streaming row by row.

Usage: python get_pages.py --count 50000 --out pages
Re-running continues after the last finished shard.
"""
import argparse, io, json, math, os, random
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "chainyo/rvl-cdip"
N_SHARDS = 119
# Label order used by chainyo/rvl-cdip (alphabetical, see the dataset card)
CLASSES = ["advertisement", "budget", "email", "file folder", "form", "handwritten", "invoice",
           "letter", "memo", "news article", "presentation", "questionnaire", "resume",
           "scientific publication", "scientific report", "specification"]
# Classes that often contain real handwriting/signatures. They would appear as
# unlabelled signatures in the training data, so they are skipped.
SKIP = {"letter", "form", "email", "handwritten", "questionnaire", "resume", "memo"}
KEEP = [c for c in CLASSES if c not in SKIP]

ap = argparse.ArgumentParser()
ap.add_argument("--count", type=int, default=50000)
ap.add_argument("--out", default="pages")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
state_file = os.path.join(args.out, "_progress.json")
state = json.load(open(state_file)) if os.path.exists(state_file) else {"done": [], "count": 0, "per_class": {}}

# cap per class so every kept class is represented even if shards are sorted by label
cap = math.ceil(args.count / len(KEEP) * 1.5)
shards = list(range(N_SHARDS))
random.Random(args.seed).shuffle(shards)
rng = random.Random(args.seed + 1)

for shard in shards:
    if state["count"] >= args.count:
        break
    if shard in state["done"]:
        continue
    name = f"data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
    print(f"Downloading {name} ...", flush=True)
    path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=os.path.join(args.out, "_tmp"))

    count, per_class = state["count"], dict(state["per_class"])
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["image", "label"]):
        rows = batch.to_pylist()
        rng.shuffle(rows)
        for row in rows:
            label = CLASSES[row["label"]]
            if label in SKIP or per_class.get(label, 0) >= cap:
                continue
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
            img.save(os.path.join(args.out, f"page_{count:05d}.jpg"), quality=92)
            per_class[label] = per_class.get(label, 0) + 1
            count += 1
            if count >= args.count:
                break
        if count >= args.count:
            break

    os.remove(path)
    state = {"done": state["done"] + [shard], "count": count, "per_class": per_class}
    json.dump(state, open(state_file, "w"))
    print(f"{count} / {args.count} pages", flush=True)

print("Finished:", state["count"], "pages")
print("Per class:", state["per_class"])
