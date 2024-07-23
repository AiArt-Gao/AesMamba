from torchvision import transforms
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torch

# IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
# IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class ParaDataset_for_add_attr(Dataset):
    def __init__(self, path_to_csv, images_path, if_train=True):
        self.csv_path = path_to_csv
        self.if_train = if_train
        self.userInfo_path = '/data/sjq/IAAdataset/PARA/annotation/PARA-UserInfo.csv'
        self.userInfoDf = pd.read_csv(self.userInfo_path)
        self.df = pd.read_csv(self.csv_path)

        self.images_path = images_path
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=BICUBIC),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        aesthetic_score = row['aestheticScore'].astype('float32')
        quality_score = row['qualityScore'].astype('float32')
        composition_score = row['compositionScore'].astype('float32')
        color_score = row['colorScore'].astype('float32')
        dof_score = row['dofScore'].astype('float32')
        light_score = row['lightScore'].astype('float32')
        content_score = row['contentScore'].astype('float32')
        contentPreference_score = row['contentPreference'].astype('float32')
        willingnessToShare_score = row['willingnessToShare'].astype('float32')
        self.imgEmotion, self.difficultyOfJudgment, self.contentPreference, self.willingnessToShare, self.semantic = \
            row['imgEmotion'], str(row['difficultyOfJudgment']), str(row['contentPreference']), str(row['willingnessToShare']), row['semantic']

        session_id, image_name, userId  = row['sessionId'], row['imageName'], row['userId']
        userInfo = self.userInfoDf[self.userInfoDf['userId'] == userId].iloc[0]
        self.age, self.gender, self.education, self.art_exp, self.photography_exp, self.E, self.A, self.N, self.O, self.C = \
            userInfo['age'], userInfo['gender'], userInfo['EducationalLevel'], userInfo['artExperience'],\
            userInfo['photographyExperience'], userInfo['personality-E'], userInfo['personality-A'], \
            userInfo['personality-N'], userInfo['personality-O'], userInfo['personality-C']

        image_path = os.path.join(self.images_path, session_id, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.if_train:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        score_dict = {'aesthetic': aesthetic_score, 'quality': quality_score, 'composition': composition_score,
                'color': color_score, 'dof': dof_score, 'light': light_score, 'content': content_score,
                'contentPreference': contentPreference_score, 'willingToShare': willingnessToShare_score}

        return image, self.get_template_text_photo_art_personality(), score_dict

    def get_template_text_photo_art_personality(self):
        template_text = f'In the Big-Five personality traits test, my scores are as follows: Openness score is {self.O},' \
                        f' conscientiousness score is {self.C}, extroversion score is {self.E}, agreeableness score is {self.A}, ' \
                        f'and Neuroticism score is {self.N}.My artistic experience is {self.art_exp}, ' \
                        f'while my photographic experience is {self.photography_exp}.'
        return template_text

