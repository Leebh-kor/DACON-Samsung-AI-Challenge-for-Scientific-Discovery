import os
import cv2
import torch
import numpy as np
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Train_Dataset(Dataset):
    def __init__(self, df, img_path, transform=None):
        df['S1-T1'] = df['S1_energy(eV)'] - df['T1_energy(eV)']
        self.uid = df['uid'].values
        self.target = df[['S1_energy(eV)', 'T1_energy(eV)', 'S1-T1']].values   # S1, T1, S1-T1
        self.img_path = img_path
        self.transform = transform

        print(f'Dataset size:{len(self.uid)}')

    def __getitem__(self, idx):
        image = cv2.imread(opj(self.img_path, self.uid[idx] + '.png')).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        target = self.target[idx]

        if self.transform is not None:
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image, target

    def __len__(self):
        return len(self.uid)

class Test_dataset(Dataset):
    def __init__(self, df, img_path, transform=None):
        self.uid = df['uid'].values
        self.img_path = img_path
        self.transform = transform

        print(f'Test Dataset size:{len(self.uid)}')

    def __getitem__(self, idx):
        image = cv2.imread(opj(self.img_path, self.uid[idx] + '.png')).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        if self.transform is not None:
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image

    def __len__(self):
        return len(self.uid)

def get_loader(df, img_path, phase: str, batch_size, shuffle,
               num_workers, transform):
    if phase == 'test':
        dataset = Test_dataset(df, img_path, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dataset = Train_Dataset(df, img_path, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                                 drop_last=False)
    return data_loader

def get_train_augmentation(img_size, ver):
    if ver==1:
        transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])

    if ver==2:
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    if ver==3:
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomErasing(scale=(0.02, 0.12), ratio=(0.3, 3.3), value=1),
	            transforms.Resize((img_size, img_size)),
    	        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    if ver==4:
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
	            transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            
    return transform

