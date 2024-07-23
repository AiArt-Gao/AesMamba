from torchvision import transforms
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class TANDataset(Dataset):
    def __init__(self, path_to_csv, images_path,isTrain,cls_num):
        self.csv_path = path_to_csv
        self.df = pd.read_csv(self.csv_path)
        self.images_path = images_path
        self.isTrain = isTrain
        self.cls_num = cls_num
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=BICUBIC),
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['score'].astype('float32')#[0,10]
        


        cls = min(math.floor(score/2),4)

        image_name = row['image']
        image_path = os.path.join(self.images_path, image_name)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.isTrain:
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, score,cls

