from torchvision import transforms
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

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


class ParaDataset_for_multi_attr(Dataset):
    def __init__(self, path_to_csv, images_path, isTrain=True):
        self.csv_path = path_to_csv
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
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.isTrain = isTrain

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        aesthetic_score = [row['aestheticScore_1.0'], row['aestheticScore_1.5'], row['aestheticScore_2.0'],
                 row['aestheticScore_2.5'], row['aestheticScore_3.0'], row['aestheticScore_3.5'],
                 row['aestheticScore_4.0'], row['aestheticScore_4.5'], row['aestheticScore_5.0']]
        y = np.array([int(k) for k in aesthetic_score]).astype('float32')
        aesthetic_p = y / y.sum()

        quality_score = row['qualityScore_mean']

        composition_score = [row['compositionScore_1'], row['compositionScore_2'], row['compositionScore_3'], row['compositionScore_4'], row['compositionScore_5']]
        y = np.array([int(k) for k in composition_score]).astype('float32')
        composition_p = y / y.sum()

        color_score = [row['colorScore_1'], row['colorScore_2'], row['colorScore_3'], row['colorScore_4'], row['colorScore_5']]
        y = np.array([int(k) for k in color_score]).astype('float32')
        color_p = y / y.sum()

        dof_score = [row['dofScore_1'], row['dofScore_2'], row['dofScore_3'], row['dofScore_4'], row['dofScore_5']]
        y = np.array([int(k) for k in dof_score]).astype('float32')
        dof_p = y / y.sum()

        light_score = [row['lightScore_1'], row['lightScore_2'], row['lightScore_3'], row['lightScore_4'], row['lightScore_5']]
        y = np.array([int(k) for k in light_score]).astype('float32')
        light_p = y / y.sum()

        content_score = [row['contentScore_1'], row['contentScore_2'], row['contentScore_3'], row['contentScore_4'],
                       row['contentScore_5']]
        y = np.array([int(k) for k in content_score]).astype('float32')
        content_p = y / y.sum()

        contentPreference_score = [row['contentPreference_1'], row['contentPreference_2'], row['contentPreference_3'],
                                   row['contentPreference_4'], row['contentPreference_5']]
        y = np.array([int(k) for k in contentPreference_score]).astype('float32')
        contentPreference_p = y / y.sum()

        willingnessToShare_score = [row['willingnessToShare_1'], row['willingnessToShare_2'], row['willingnessToShare_3'],
                                    row['willingnessToShare_4'], row['willingnessToShare_5']]
        y = np.array([int(k) for k in willingnessToShare_score]).astype('float32')
        willingnessToshare_p = y / y.sum()

        session_id = row['sessionId']
        image_name = row['imageName']
        image_path = os.path.join(self.images_path, session_id, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.isTrain:
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, {'aesthetic': aesthetic_p, 'quality': quality_score.astype('float32'), 'composition': composition_p,
                'color': color_p, 'dof': dof_p, 'light': light_p, 'content': content_p,
                'contentPreference': contentPreference_p, 'willingToShare': willingnessToshare_p}
        # return x, p.astype('float16')
               # aesthetic.astype('float32'), quality.astype('float32'), composition.astype('float32'), \
               # color.astype('float32'), dof.astype('float32'), light.astype('float32'), content.astype('float32')
        # return x, y.astype('float32'),image_path,coordinate

class ParaDataset_for_multi_attr_and_classification(Dataset):
    def __init__(self, path_to_csv, images_path, isTrain=True):
        self.csv_path = path_to_csv
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
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.isTrain = isTrain

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        aesthetic_score = [row['aestheticScore_1.0'], row['aestheticScore_1.5'], row['aestheticScore_2.0'],
                 row['aestheticScore_2.5'], row['aestheticScore_3.0'], row['aestheticScore_3.5'],
                 row['aestheticScore_4.0'], row['aestheticScore_4.5'], row['aestheticScore_5.0']]
        y = np.array([int(k) for k in aesthetic_score]).astype('float32')
        aesthetic_p = y / y.sum()

        quality_score = row['qualityScore_mean']

        composition_score = [row['compositionScore_1'], row['compositionScore_2'], row['compositionScore_3'], row['compositionScore_4'], row['compositionScore_5']]
        y = np.array([int(k) for k in composition_score]).astype('float32')
        composition_p = y / y.sum()

        color_score = [row['colorScore_1'], row['colorScore_2'], row['colorScore_3'], row['colorScore_4'], row['colorScore_5']]
        y = np.array([int(k) for k in color_score]).astype('float32')
        color_p = y / y.sum()

        dof_score = [row['dofScore_1'], row['dofScore_2'], row['dofScore_3'], row['dofScore_4'], row['dofScore_5']]
        y = np.array([int(k) for k in dof_score]).astype('float32')
        dof_p = y / y.sum()

        light_score = [row['lightScore_1'], row['lightScore_2'], row['lightScore_3'], row['lightScore_4'], row['lightScore_5']]
        y = np.array([int(k) for k in light_score]).astype('float32')
        light_p = y / y.sum()

        content_score = [row['contentScore_1'], row['contentScore_2'], row['contentScore_3'], row['contentScore_4'],
                       row['contentScore_5']]
        y = np.array([int(k) for k in content_score]).astype('float32')
        content_p = y / y.sum()

        contentPreference_score = [row['contentPreference_1'], row['contentPreference_2'], row['contentPreference_3'],
                                   row['contentPreference_4'], row['contentPreference_5']]
        y = np.array([int(k) for k in contentPreference_score]).astype('float32')
        contentPreference_p = y / y.sum()

        willingnessToShare_score = [row['willingnessToShare_1'], row['willingnessToShare_2'], row['willingnessToShare_3'],
                                    row['willingnessToShare_4'], row['willingnessToShare_5']]
        y = np.array([int(k) for k in willingnessToShare_score]).astype('float32')
        willingnessToshare_p = y / y.sum()

        aesthetic_class = round(row['aestheticScore_mean'])-1
        quality_class = round(row['qualityScore_mean'])-1
        composition_class = round(row['compositionScore_mean'])-1
        color_class = round(row['colorScore_mean'])-1
        dof_class = round(row['dofScore_mean'])-1
        light_class = round(row['lightScore_mean'])-1
        content_class = round(row['contentScore_mean'])-1
        contentPreference_class = round(row['contentPreference_mean'])-1
        willingnessToShare_class = round(row['willingnessToShare_mean'])-1

        session_id = row['sessionId']
        image_name = row['imageName']
        image_path = os.path.join(self.images_path, session_id, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.isTrain:
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, {'aesthetic': aesthetic_p, 'quality': quality_score.astype('float32'), 'composition': composition_p,
                'color': color_p, 'dof': dof_p, 'light': light_p, 'content': content_p,
                'contentPreference': contentPreference_p, 'willingToShare': willingnessToshare_p}, \
               {'aesthetic': aesthetic_class, 'quality': quality_class, 'composition': composition_class,
                'color': color_class, 'dof': dof_class, 'light': light_class, 'content': content_class,
                'contentPreference': contentPreference_class, 'willingToShare': willingnessToShare_class}
        # return x, p.astype('float16')
               # aesthetic.astype('float32'), quality.astype('float32'), composition.astype('float32'), \
               # color.astype('float32'), dof.astype('float32'), light.astype('float32'), content.astype('float32')
        # return x, y.astype('float32'),image_path,coordinate

class ParaDataset_for_multi_attr_for_mse(Dataset):
    def __init__(self, path_to_csv, images_path, isTrain=True):
        self.csv_path = path_to_csv
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
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.isTrain = isTrain

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        aesthetic_score = row['aestheticScore_mean'].astype('float32')
        quality_score = row['qualityScore_mean'].astype('float32')
        composition_score = row['compositionScore_mean'].astype('float32')
        color_score = row['colorScore_mean'].astype('float32')
        dof_score = row['dofScore_mean'].astype('float32')
        light_score = row['lightScore_mean'].astype('float32')
        content_score = row['contentScore_mean'].astype('float32')
        contentPreference_score = row['contentPreference_mean'].astype('float32')
        willingnessToShare_score = row['willingnessToShare_mean'].astype('float32')

        session_id = row['sessionId']
        image_name = row['imageName']
        image_path = os.path.join(self.images_path, session_id, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.isTrain:
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, {'aesthetic': aesthetic_score, 'quality': quality_score, 'composition': composition_score,
                'color': color_score, 'dof': dof_score, 'light': light_score, 'content': content_score,
                'contentPreference': contentPreference_score, 'willingToShare': willingnessToShare_score}
        # return x, p.astype('float16')
               # aesthetic.astype('float32'), quality.astype('float32'), composition.astype('float32'), \
               # color.astype('float32'), dof.astype('float32'), light.astype('float32'), content.astype('float32')
        # return x, y.astype('float32'),image_path,coordinate


class ParaDataset_for_multi_attr_and_4class(Dataset):
    def __init__(self, path_to_csv, images_path, isTrain=True):
        self.csv_path = path_to_csv
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
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.isTrain = isTrain

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        aesthetic_score = [row['aestheticScore_1.0'], row['aestheticScore_1.5'], row['aestheticScore_2.0'],
                 row['aestheticScore_2.5'], row['aestheticScore_3.0'], row['aestheticScore_3.5'],
                 row['aestheticScore_4.0'], row['aestheticScore_4.5'], row['aestheticScore_5.0']]
        y = np.array([int(k) for k in aesthetic_score]).astype('float32')
        aesthetic_p = y / y.sum()

        quality_score = row['qualityScore_mean']

        composition_score = [row['compositionScore_1'], row['compositionScore_2'], row['compositionScore_3'], row['compositionScore_4'], row['compositionScore_5']]
        y = np.array([int(k) for k in composition_score]).astype('float32')
        composition_p = y / y.sum()

        color_score = [row['colorScore_1'], row['colorScore_2'], row['colorScore_3'], row['colorScore_4'], row['colorScore_5']]
        y = np.array([int(k) for k in color_score]).astype('float32')
        color_p = y / y.sum()

        dof_score = [row['dofScore_1'], row['dofScore_2'], row['dofScore_3'], row['dofScore_4'], row['dofScore_5']]
        y = np.array([int(k) for k in dof_score]).astype('float32')
        dof_p = y / y.sum()

        light_score = [row['lightScore_1'], row['lightScore_2'], row['lightScore_3'], row['lightScore_4'], row['lightScore_5']]
        y = np.array([int(k) for k in light_score]).astype('float32')
        light_p = y / y.sum()

        content_score = [row['contentScore_1'], row['contentScore_2'], row['contentScore_3'], row['contentScore_4'],
                       row['contentScore_5']]
        y = np.array([int(k) for k in content_score]).astype('float32')
        content_p = y / y.sum()

        contentPreference_score = [row['contentPreference_1'], row['contentPreference_2'], row['contentPreference_3'],
                                   row['contentPreference_4'], row['contentPreference_5']]
        y = np.array([int(k) for k in contentPreference_score]).astype('float32')
        contentPreference_p = y / y.sum()

        willingnessToShare_score = [row['willingnessToShare_1'], row['willingnessToShare_2'], row['willingnessToShare_3'],
                                    row['willingnessToShare_4'], row['willingnessToShare_5']]
        y = np.array([int(k) for k in willingnessToShare_score]).astype('float32')
        willingnessToshare_p = y / y.sum()

        aesthetic_class = row['aestheticScore_class']
        quality_class = row['qualityScore_class']
        composition_class = row['compositionScore_class']
        color_class = row['colorScore_class']
        dof_class = row['dofScore_class']
        light_class = row['lightScore_class']
        content_class = row['contentScore_class']
        contentPreference_class = row['contentPreference_class']
        willingnessToShare_class = row['willingnessToShare_class']

        session_id = row['sessionId']
        image_name = row['imageName']
        image_path = os.path.join(self.images_path, session_id, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.isTrain:
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, {'aesthetic': aesthetic_p, 'quality': quality_score.astype('float32'), 'composition': composition_p,
                'color': color_p, 'dof': dof_p, 'light': light_p, 'content': content_p,
                'contentPreference': contentPreference_p, 'willingToShare': willingnessToshare_p}, \
               {'aesthetic': aesthetic_class, 'quality': quality_class, 'composition': composition_class,
                'color': color_class, 'dof': dof_class, 'light': light_class, 'content': content_class,
                'contentPreference': contentPreference_class, 'willingToShare': willingnessToShare_class}
        # return x, p.astype('float16')
               # aesthetic.astype('float32'), quality.astype('float32'), composition.astype('float32'), \
               # color.astype('float32'), dof.astype('float32'), light.astype('float32'), content.astype('float32')
        # return x, y.astype('float32'),image_path,coordinate

class ParaDataset_for_multi_attr_mse_and_4class(Dataset):
    def __init__(self, path_to_csv, images_path, isTrain=True):
        self.csv_path = path_to_csv
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
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.isTrain = isTrain

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        aesthetic_score = row['aestheticScore_mean'].astype('float32')
        quality_score = row['qualityScore_mean'].astype('float32')
        composition_score = row['compositionScore_mean'].astype('float32')
        color_score = row['colorScore_mean'].astype('float32')
        dof_score = row['dofScore_mean'].astype('float32')
        light_score = row['lightScore_mean'].astype('float32')
        content_score = row['contentScore_mean'].astype('float32')
        contentPreference_score = row['contentPreference_mean'].astype('float32')
        willingnessToShare_score = row['willingnessToShare_mean'].astype('float32')

        aesthetic_class = row['aestheticScore_class']
        quality_class = row['qualityScore_class']
        composition_class = row['compositionScore_class']
        color_class = row['colorScore_class']
        dof_class = row['dofScore_class']
        light_class = row['lightScore_class']
        content_class = row['contentScore_class']
        contentPreference_class = row['contentPreference_class']
        willingnessToShare_class = row['willingnessToShare_class']

        session_id = row['sessionId']
        image_name = row['imageName']
        image_path = os.path.join(self.images_path, session_id, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.isTrain:
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, {'aesthetic': aesthetic_score, 'quality': quality_score, 'composition': composition_score,
                'color': color_score, 'dof': dof_score, 'light': light_score, 'content': content_score,
                'contentPreference': contentPreference_score, 'willingToShare': willingnessToShare_score}, \
               {'aesthetic': aesthetic_class, 'quality': quality_class, 'composition': composition_class,
                'color': color_class, 'dof': dof_class, 'light': light_class, 'content': content_class,
                'contentPreference': contentPreference_class, 'willingToShare': willingnessToShare_class}
        # return x, p.astype('float16')
               # aesthetic.astype('float32'), quality.astype('float32'), composition.astype('float32'), \
               # color.astype('float32'), dof.astype('float32'), light.astype('float32'), content.astype('float32')
        # return x, y.astype('float32'),image_path,coordinate

class ParaDataset_for_multi_attr_mse_and_5class(Dataset):
    def __init__(self, path_to_csv, images_path, isTrain=True):
        self.csv_path = path_to_csv
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
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.isTrain = isTrain

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        aesthetic_score = row['aestheticScore_mean'].astype('float32')
        quality_score = row['qualityScore_mean'].astype('float32')
        composition_score = row['compositionScore_mean'].astype('float32')
        color_score = row['colorScore_mean'].astype('float32')
        dof_score = row['dofScore_mean'].astype('float32')
        light_score = row['lightScore_mean'].astype('float32')
        content_score = row['contentScore_mean'].astype('float32')
        contentPreference_score = row['contentPreference_mean'].astype('float32')
        willingnessToShare_score = row['willingnessToShare_mean'].astype('float32')

        aesthetic_class = round(row['aestheticScore_mean']) - 1
        quality_class = round(row['qualityScore_mean']) - 1
        composition_class = round(row['compositionScore_mean']) - 1
        color_class = round(row['colorScore_mean']) - 1
        dof_class = round(row['dofScore_mean']) - 1
        light_class = round(row['lightScore_mean']) - 1
        content_class = round(row['contentScore_mean']) - 1
        contentPreference_class = round(row['contentPreference_mean']) - 1
        willingnessToShare_class = round(row['willingnessToShare_mean']) - 1

        session_id = row['sessionId']
        image_name = row['imageName']
        image_path = os.path.join(self.images_path, session_id, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.isTrain:
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, {'aesthetic': aesthetic_score, 'quality': quality_score, 'composition': composition_score,
                'color': color_score, 'dof': dof_score, 'light': light_score, 'content': content_score,
                'contentPreference': contentPreference_score, 'willingToShare': willingnessToShare_score}, \
               {'aesthetic': aesthetic_class, 'quality': quality_class, 'composition': composition_class,
                'color': color_class, 'dof': dof_class, 'light': light_class, 'content': content_class,
                'contentPreference': contentPreference_class, 'willingToShare': willingnessToShare_class}
        # return x, p.astype('float16')
               # aesthetic.astype('float32'), quality.astype('float32'), composition.astype('float32'), \
               # color.astype('float32'), dof.astype('float32'), light.astype('float32'), content.astype('float32')
        # return x, y.astype('float32'),image_path,coordinate