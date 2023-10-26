import torch
import cv2
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt


IMG_DIR = './flickr8K/images/'
ANNOTATION = './flickr8K/captions.txt'

class flickr8K_dataset(Dataset):
    def __init__(self, img_dir, annotation, transforms=None, trainining=True) -> None:

        self.img_dir = img_dir
        self.annotation = annotation
        self.transforms = transforms
        self.training = trainining

        label_indices = {}
        img_indices = {}

        with open(annotation) as txt:
            lines = txt.readlines()
            self.lenlines = len(lines)-1

            for idx, line in enumerate(lines[1:]):
                
                img = line.split(",", 1)[0]
                caption = line.split(",", 1)[1].strip()

                label_indices[idx] = caption
                img_indices[idx] = img
        
        self.img_indices = img_indices
        self.label_indices = label_indices

    def __len__(self):
        return self.lenlines
    

    def __getitem__(self, index):

        
        label = self.label_indices[index]
        img_name = self.img_indices[index]

        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, label



#dataset = flickr8K_dataset(IMG_DIR, ANNOTATION)

#img = plt.imshow(dataset[0][0])
#plt.show()
        


