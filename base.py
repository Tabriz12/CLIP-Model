import torch
import cv2
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

#IMG_DIR = './flickr8K/images/'
#ANNOTATION = './flickr8K/captions.txt'

with open('./cfg.yaml') as cfg_file:

    CFG = yaml.safe_load(cfg_file)




class flickr8K_dataset(Dataset):
    def __init__(self, img_dir=CFG['dataset']['images'], 
                 annotation=CFG['dataset']['annotations'], 
                 txt_embedder=CFG['models']['text_embedder'], 
                 transforms=None, 
                 trainining=True) -> None:

        self.img_dir = img_dir
        self.annotation = annotation
        self.txt_embedder = txt_embedder
        self.transforms = transforms
        self.training = trainining

        #label_indices = {}
        embedding_indices = {}
        img_indices = {}
        embedding_model = SentenceTransformer(txt_embedder, device='cuda')
        with open(annotation) as txt:
            lines = txt.readlines()
            self.lenlines = len(lines)-1

            for idx, line in tqdm(enumerate(lines[1:])):
                
                img = line.split(",", 1)[0]
                caption = line.split(",", 1)[1].strip()

                embedding_indices[idx] = embedding_model.encode(caption)
                #label_indices[idx] = caption
                img_indices[idx] = img
        
        self.img_indices = img_indices
        #self.label_indices = label_indices
        self.embedding_indices = embedding_indices

    def __len__(self):
        return self.lenlines
    

    def __getitem__(self, index):

        
        #label = self.label_indices[index]
        img_name = self.img_indices[index]
        txt_embedding = self.embedding_indices[index]

        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, txt_embedding



dataset = flickr8K_dataset()
print(dataset[0][2])
print(dataset[0][1])
#img = plt.imshow(dataset[0][0])
#plt.show()
        


