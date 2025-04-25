"""
Lightning trainer – fine-tunes projector + final Llama layers on VQAv2.
"""
import random
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T

from ..models.multimodal_model import InSightAIModel
from ..config import DATA_DIR, BATCH, EPOCHS, LR

# Ensure cache dir exists
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

class VQADataset(Dataset):
    def __init__(self, split: str = "train"):
        self.ds = load_dataset(
            "HuggingFaceM4/VQAv2",
            split=split,
            cache_dir=str(DATA_DIR),
        )

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        # Load & preprocess image
        img = (
            Image.open(row["image_path"])
            .convert("RGB")
            .resize((224, 224))
        )
        question = row["question"]
        answer   = random.choice(row["answers"])
        return img, question, answer

    def __len__(self) -> int:
        return len(self.ds)

class LitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mm = InSightAIModel()
        self.transform = T.ToTensor()
        # Save all hyperparameters except non-serializable ones
        self.save_hyperparameters(ignore=["mm", "transform"])

    def training_step(self, batch, batch_idx):
        imgs, qs, ans = zip(*batch)
        # PIL Image → Tensor → GPU
        imgs = torch.stack([self.transform(img) for img in imgs]).to(self.device)
        loss = self.mm(imgs, list(qs), list(ans))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, qs, ans = zip(*batch)
        imgs = torch.stack([self.transform(img) for img in imgs]).to(self.device)
        loss = self.mm(imgs, list(qs), list(ans))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.mm.parameters(), lr=LR)

def run():
    train_ds = VQADataset("train")
    val_ds   = VQADataset("validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=EPOCHS,
        precision="bf16",
        log_every_n_steps=10,
    )
    trainer.fit(LitModule(), train_loader, val_loader)

if __name__ == "__main__":
    run()
