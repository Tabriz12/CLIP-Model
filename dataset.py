import torch
import cv2
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import CFG
from transformers import DistilBertTokenizer


class flickr8K_dataset(Dataset):
    def __init__(self, img_dir=CFG['dataset']['images'], 
                 annotation=CFG['dataset']['annotations'], 
                 txt_embedder=CFG['models']['text_embedder'], 
                 transforms=None, 
                 ) -> None:

        self.img_dir = img_dir
        self.annotation = annotation
        self.txt_embedder = txt_embedder
        self.transforms = transforms
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        

        label_indices = {}
        img_indices = {}
        captions = []
        
        with open(annotation) as txt:
            lines = txt.readlines()

            for idx, line in tqdm(enumerate(lines[1:])):
                
                img = line.split(",", 1)[0]
                caption = line.split(",", 1)[1].strip()

                label_indices[idx] = caption
                img_indices[idx] = img
                captions.append(caption)
                
        
        self.img_indices = img_indices
        self.label_indices = label_indices
        self.tokenized = tokenizer(captions, truncation=True, padding=True, return_tensors="pt")
        



    def __len__(self):
        return len(self.img_indices)
    

    def __getitem__(self, index):

        
        label = {'input_ids':self.tokenized['input_ids'][index],
                 'attention_mask':self.tokenized['attention_mask'][index]
                 }
        img_name = self.img_indices[index]
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, label



dataset = flickr8K_dataset()
print(dataset[0][1])
#(len(dataset))
#img = plt.imshow(dataset[0][0])
#plt.show()


        


