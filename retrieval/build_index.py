import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from opensearchpy import OpenSearch, helpers
from ..models.vision_encoder import VisionEncoder
from ..config import DATA_DIR, OPENSEARCH as OS

# Ensure cache dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Connect to OpenSearch
client = OpenSearch(
    hosts=[OS["hosts"]],
    http_auth=(OS["user"], OS["password"]),
    timeout=120,
)

# Create index with k-NN mapping if missing
if not client.indices.exists(index=OS["index"]):
    client.indices.create(
        index=OS["index"],
        body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "image_id": {"type": "keyword"},
                    "question": {"type": "text"},
                    "answers":  {"type": "text"},
                    "vector":   {
                        "type":      "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name":       "hnsw",
                            "space_type": "l2",
                            "engine":     "nmslib",
                            "parameters": {"ef_construction": 128, "m": 24},
                        },
                    },
                }
            },
        },
    )

# Load VQAv2 dataset
ds = load_dataset("HuggingFaceM4/VQAv2", split="train", cache_dir=str(DATA_DIR))

# Embed & bulk‐upload
vit = VisionEncoder().eval().cuda()

def gen_docs():
    for row in tqdm(ds, desc="Indexing images"):
        # load & preprocess
        img = Image.open(row["image_path"]).convert("RGB").resize((224, 224))
        with torch.no_grad():
            vec = vit(img.unsqueeze(0).cuda()).cpu().numpy()[0]
        yield {
            "_index": OS["index"],
            "_id":    row["image_id"],
            "_source": {
                "image_id": row["image_id"],
                "question": row["question"],
                "answers":  " | ".join(row["answers"]),
                "vector":   vec.tolist(),
            },
        }

helpers.bulk(client, gen_docs(), chunk_size=256, request_timeout=120)
print(f"OpenSearch index '{OS['index']}' built with {len(ds)} vectors.")
