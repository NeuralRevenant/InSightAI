from typing import Tuple
import torch
from transformers import CLIPProcessor, CLIPModel
from ..config import VIT_NAME, IMG_SIZE, HF_CACHE

class VisionEncoder(torch.nn.Module):
    """
    Wrapper that returns a (B, N, D) tensor where N=token count and D=768 for ViT-L/14.
    """
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(VIT_NAME, cache_dir=HF_CACHE)
        self.processor = CLIPProcessor.from_pretrained(VIT_NAME, cache_dir=HF_CACHE)
        self.token_dim = self.clip.vision_model.config.hidden_size   # 1024

    @torch.no_grad()
    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(images.device)
        return self.clip.get_image_features(**inputs)          # (B, 1024)
