import os
import re
from torchvision import transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import ImageFile, Image
# from transform.randaugment import RandomAugment
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = T.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AVA_Comment_Dataset_bert_balce(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                normalize])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                # T.CenterCrop((224, 224)),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        # score = row['score']
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()
        cls = row['class']

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)



        return image,  p.astype('float32'), cls

