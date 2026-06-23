import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
from PIL import Image

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    return device


class BananaDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.csv_path = os.path.join(root_dir, 'label.csv')
        self.annotations = pd.read_csv(self.csv_path)
        self.transoform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, indcies):
        row = self.annotations.iloc[indcies]
        images_name = row['img_name']
        images_path = os.path.join(self.images_dir,images_name)
        images = Image.open(images_path).convert('RGB')
        images_tensor = self.transoform(images)

        boxes = torch.tensor([
            [
                row['xmin'],
                row['ymin'],
                row['xmax'],
                row['ymax']
            ]
        ],dtype=torch.float32
        )

        labels = torch.tensor(
            [row['label'] + 1],
            dtype=torch.long
        )

        targets = {
            'boxes': boxes,
            'labels': labels,
        }

        return images_tensor,targets

def collate_fn(batch):
    images, targets = zip(*batch)

    return torch.stack(images), list(targets)


def get_train_dataloader(root_dir,
                         batch_size,
                         shuffle = True,
                         num_workers = 6):
    dataset = BananaDataset(root_dir)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        collate_fn = collate_fn
    )
    return dataloader


def get_test_dataloader(root_dir,
                        batch_size,
                        shuffle = True,
                        num_workers = 6):
    dataset = BananaDataset(root_dir)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        collate_fn = collate_fn
    )
    return dataloader