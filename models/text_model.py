from torch import nn
from config import CFG
from transformers import AlbertModel

class TextModel(nn.Module):
    def __init__(self ) -> None:
        super().__init__()
        self.model = AlbertModel.from_pretrained('')
        


    

    def forward(self, x):
        pass
