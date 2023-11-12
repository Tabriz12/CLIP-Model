import torch
from torch import nn
from models.text_model import TextModel
from models.feature_mapper import FeatureMapper
from models.vision_model import VisionModel

class CLIP(nn.Module):
    def __init__(self, 
                 dim_img: int = 768, 
                 dim_txt: int = 768,
                 dim_out: int = 256,
                 temperature = 2.0):
        super().__init__()
        self.vision_model = VisionModel()
        self.text_model = TextModel()
        self.img_embbedder = FeatureMapper(dim_img, dim_out)
        self.txt_embedder = FeatureMapper(dim_txt, dim_out)
        self.temperature = temperature
        #self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=True) # network parameter as well

    
    def forward(self, image, text):

        vision_out = self.vision_model(image)

        text_out = self.text_model(text)

        img_emb = self.img_embbedder(vision_out)

        txt_emb = self.txt_embedder(text_out)

        logits_txt = (txt_emb @ img_emb.T) / self.temperature
        logits_img = (img_emb @ txt_emb.T) / self.temperature

        loss_txt = nn.functional.cross_entropy(logits_txt, torch.arange(len(logits_txt), device=logits_txt.device))
        loss_img = nn.functional.cross_entropy(logits_img, torch.arange(len(logits_img), device=logits_img.device))
        final_loss = (loss_txt + loss_img) / 2.0
        return final_loss

        


        


