import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import pytorch_lightning as pl
# from src.models.components.clip import CLIP
from typing import Dict, List, Tuple
from transformers import AutoModel, AutoTokenizer, BertTokenizer


class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=self.hparams.max_len, truncation=True, padding="max_length", return_tensors="pt"
        )

    def decode(self, x: Dict[str, torch.LongTensor]):
        return [self.tokenizer.decode(sentence[:sentence_len]) for sentence, sentence_len in 
                zip(x["input_ids"], x["attention_mask"].sum(axis=-1))]

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisionEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        base = models.resnet34(pretrained=True)
        d_in = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.projection = Projection(d_in, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

class TextEncoder(nn.Module):
    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(self.hparams.text_model)
        self.projection = Projection(self.hparams.transformer_embed_dim, d_out)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

def metrics(similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc

class CLIPModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.vision_encoder = VisionEncoder(self.hparams.embed_dim)
        self.caption_encoder = TextEncoder(self.hparams.embed_dim)
        self.tokenizer = Tokenizer(AutoTokenizer.from_pretrained(self.hparams.text_model))
        self.lr = lr
       
    def common_step(self, batch: Tuple[torch.Tensor, List[str]]) -> torch.Tensor:
        images, text = batch
        device = images.device
        text_dev = {k: v.to(device) for k, v in self.tokenizer(text).items()}
        image_embed = self.vision_encoder(images)
        caption_embed = self.caption_encoder(text_dev)
        similarity = caption_embed @ image_embed.T
        loss = clip_loss(similarity)
        img_acc, cap_acc = metrics(similarity)
        return loss, img_acc, cap_acc

    def training_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, cap_acc = self.common_step(batch)     
        self.log("training_loss", loss, on_step=True)
        self.log("training_img_acc", img_acc, on_step=True, prog_bar=True)
        self.log("training_cap_acc", cap_acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("validation_loss", loss, on_step=True)
        self.log("validation_img_acc", img_acc, on_step=True, prog_bar=True)
        self.log("validation_cap_acc", cap_acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        vision_params = {"params": self.vision_encoder.projection.parameters(), "lr": self.lr}
        caption_params = {"params": self.caption_encoder.projection.parameters() , "lr": self.lr}
        return torch.optim.Adam([vision_params, caption_params])