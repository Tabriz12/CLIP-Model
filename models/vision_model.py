import torch
from torch import nn
import torch.nn.functional as F

from transformers import ViTModel


class VisionModel(nn.Module):
    def __init__(self, training = True ) -> None:
        super().__init__()


        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        if training: self.vit_model.train()
        else: self.vit_model.eval()
        
    
    
    def forward(self, x):

        self.vit_model(x)
