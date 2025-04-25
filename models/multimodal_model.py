import torch, torch.nn as nn
from .vision_encoder import VisionEncoder
from .language_model import LanguageModel
from .projector import Projector

class InSightAIModel(nn.Module):
    """
    LLaVA-like fusion: <magic_token> + Vision-features-as-tokens + question tokens → LLM
    """
    VISION_TOKEN = "<|vision|>"

    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.llm    = LanguageModel()
        self.proj   = Projector(
            in_dim=self.vision.token_dim, out_dim=self.llm.embed_dim
        )
        # Add vision special token to tokenizer
        if self.VISION_TOKEN not in self.llm.tokenizer.get_vocab():
            self.llm.tokenizer.add_tokens([self.VISION_TOKEN])
            self.llm.model.resize_token_embeddings(len(self.llm.tokenizer))

    def forward(self, images, questions, answers=None):
        device = next(self.parameters()).device
        bsz = images.size(0)
        # Encode image
        vis_emb = self.vision(images)                # (B, 1024)
        vis_emb = self.proj(vis_emb)                 # (B, 4096)
        # Convert to 1 token by prepending to prompt
        vision_tokens = torch.tensor(
            [self.llm.tokenizer.convert_tokens_to_ids(self.VISION_TOKEN)], device=device
        ).repeat(bsz, 1)
        # Tokenize question
        q_tok = self.llm.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        input_ids = torch.cat([vision_tokens, q_tok["input_ids"]], dim=1)
        # Map projected vision feature to embedding matrix
        self.llm.model.get_input_embeddings().weight[
            vision_tokens[0, 0]
        ] = vis_emb.mean(dim=0)  # quick trick: overwrite the token vector

        if answers is None:                   # Inference
            outputs = self.llm.generate(
                input_ids,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.2,
                top_p=0.9
            )
            return self.llm.tokenizer.batch_decode(
                outputs[:, input_ids.shape[1] :], skip_special_tokens=True
            )
        else:                                 # Training
            with self.llm.tokenizer.as_target_tokenizer():
                labels = self.llm.tokenizer(
                    answers, return_tensors="pt", padding=True, truncation=True
                ).input_ids.to(device)
            out = self.llm.model(input_ids=input_ids, labels=labels)
            return out.loss
