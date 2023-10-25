import torch
import cv2
from torch.utils.data import Dataset
import os


class flickr8K_dataset(Dataset):
    def __init__(self, root, labels, transforms, train=True) -> None:

        self.root = root
        self.labels = labels
        self.transforms = transforms
        self.train = train
    

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
    
    def __len__(self):
        return os.listdir(len(self.root))

        


