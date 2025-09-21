import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from typing import Optional


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir, num_datapoints: Optional[int] = None, num_data: Optional[int] = None):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        # self.flip =  False # 'flip' in self.feature_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files = sorted(os.listdir(feature_dir))
        self.label_files = sorted(os.listdir(label_dir))
        if num_datapoints is not None:
            self.feature_files = [self.feature_files[i%num_datapoints] for i in range(len(self.feature_files))] # sample only num_datapoints files
            self.label_files = [self.label_files[i%num_datapoints] for i in range(len(self.label_files))] # sample only num_datapoints files
        
        if num_data is not None:
            self.feature_files = self.feature_files[:num_data] # sample only first num_data files
            self.label_files = self.label_files[:num_data] # sample only first num_data files
                
    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir

        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        else:
            features = features[:, 0]
            
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    num_datapoints = args.num_datapoints
    #############################################
    num_data = args.num_data
    #############################################
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir, num_datapoints, num_data)