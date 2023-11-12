import torch
import cv2
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import CFG
from transformers import AlbertTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
class Flickr8K_dataset(Dataset):
    def __init__(self, img_dir=CFG['dataset']['images'], 
                 annotation=CFG['dataset']['annotations'], 
                 tokenizer = tokenizer,
                 split = 'train',
                 transform = None,
                 split_ratio = 4,
                 ) -> None:
        

        self.img_dir = img_dir
        self.transform = transform
        if split not in ['train', 'test']:
            raise ValueError('split value can be either train or test')
        
        img_names = []
        captions = []
        
        
        with open(annotation) as txt:
            lines = txt.readlines()

            for idx, line in enumerate(lines[1:]):

                if (((split == 'train') and (idx % (split_ratio+1) != 0)) or ((split == 'test') and (idx % (split_ratio+1) == 0))):


                    img = line.split(",", 1)[0]
                    caption = line.split(",", 1)[1].strip()

                    img_names.append(img)
                    captions.append(caption)
                
        
        self.img_indices = img_names
        self.tokenized = tokenizer(captions, truncation=True, return_tensors="pt", padding = True)


    def __len__(self):
        return len(self.img_indices)
    

    def __getitem__(self, index):
        
        caption = {'input_ids':self.tokenized['input_ids'][index].to(CFG['device']),
                 'attention_mask':self.tokenized['attention_mask'][index].to(CFG['device'])
                 } #'token_type_ids': self.tokenized['token_type_ids'][index],
        
        img_name = self.img_indices[index]
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        if self.transform is not None:
            image = self.transform(image=image)["image"].to(CFG['device'])
        
        return image, caption


train_transform = A.Compose(
    [
        A.Resize(224, 224, always_apply = True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(224, 224, always_apply = True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

        


