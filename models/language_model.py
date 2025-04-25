from transformers import AutoModelForCausalLM, AutoTokenizer
from ..config import LLM_NAME, HF_CACHE
import torch

class LanguageModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME, cache_dir=HF_CACHE, padding_side="left", use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, cache_dir=HF_CACHE, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.embed_dim = self.model.config.hidden_size   # 4096 for Llama-3-8B

    def generate(self, input_ids, **kw):
        return self.model.generate(input_ids=input_ids, **kw)
