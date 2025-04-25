"""
Pre-downloads VQAv2 images & annotations into data/cache/
before training the following is run:  python -m insightai.data.prepare_vqa
"""
from datasets import load_dataset
from ..config import DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
ds = load_dataset("HuggingFaceM4/VQAv2", cache_dir=str(DATA_DIR))
print(ds)
