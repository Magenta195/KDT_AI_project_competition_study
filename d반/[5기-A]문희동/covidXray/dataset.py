import torch
import pandas as pd
import os

from PIL import Image


class ImageFileCsvLabelDataset(torch.utils.data.Dataset): 
    def __init__(self, csvfile_path ,imgfile_path, option=None, transform=None):
        self.csv_label = pd.read_csv(csvfile_path)
        self.imgfile_path = imgfile_path
        self.transform = transform
        self.option = option

    def __len__(self):
        return len(self.csv_label)
    
    def __getitem__(self, idx):
        file_name  = self.csv_label.iloc[idx,0]
        image=  Image.open(os.path.join(self.imgfile_path, file_name)).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.option =='train':
            label = self.csv_label.iloc[idx,1]
            if label == 'covid':
                label = int(1)
            else:
                label = int(0)
            label = torch.tensor(label, dtype=torch.int64)
            return image, label
        
        return image