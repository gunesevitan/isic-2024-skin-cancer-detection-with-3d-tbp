import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import geffnet

sys.path.append('..')
import settings


class SIIMISICDataset(Dataset):

    def __init__(self, csv, split, mode, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.image_path)
        image = image[:, :, ::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.mode == 'test':
            return torch.tensor(image).float()
        else:
            return torch.tensor(image).float(), torch.tensor(self.csv.iloc[index].target).long()


class enetv2(nn.Module):

    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=False):

        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=load_pretrained)
        self.dropout = nn.Dropout(0.5)

        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        x = self.myfc(self.dropout(x))
        return x


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)



if __name__ == '__main__':

    model_directory = settings.MODELS / 'isic_2020_winning_models'

    device = torch.device('cuda')

    kernel_type = '9c_b4ns_2e_896_ext_15ep'
    image_size = 640
    use_amp = True
    enet_type = 'efficientnet-b4'
    out_dim = 9

    use_external = '_ext' in kernel_type

    df_metadata = pd.read_parquet(settings.DATA / 'isic_master_dataset' / 'metadata.parquet')
    df_metadata = df_metadata.loc[df_metadata['dataset'] == 'isic_2024'].reset_index()

    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()
    ])

    n_test = 8
    dataset_test = SIIMISICDataset(df_metadata, 'test', 'test', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, num_workers=16)

    models = []
    for i_fold in range(5):
        model = enetv2(enet_type, n_meta_features=0, out_dim=out_dim)
        model = model.to(device)
        model_file = os.path.join(model_directory, f'{kernel_type}_best_fold{i_fold}.pth')
        state_dict = torch.load(model_file)
        state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)

    OUTPUTS = []
    PROBS = []

    with torch.no_grad():
        for (data) in tqdm(test_loader):

            data = data.to(device)
            probs = torch.zeros((data.shape[0], out_dim)).to(device)
            for model in models:
                for I in range(n_test):
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        l = model(get_trans(data, I))
                    probs += l.softmax(1)

            probs /= n_test * len(models)
            PROBS.append(probs.detach().cpu())

    PROBS = torch.cat(PROBS).numpy()
    OUTPUTS = PROBS[:, 6]
