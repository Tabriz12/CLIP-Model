from torch import nn
from transformers import AlbertModel
import torch

class TextModel(nn.Module):
    def __init__(self, training=True):

        super().__init__()
        self.model = AlbertModel.from_pretrained('albert-base-v2')
        self.model.train(training)
    

    def forward(self, x):
        out = self.model(**x)
        return out.pooler_output

