import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
import os
import cv2
import numpy as np

def one_hot(label, n_class):
    """
    :param label: has shape (H, W)
    :param n_class:
    :return: shape (n_class, H, W)
    """
    hot = np.zeros(label.shape + (n_class,))        # (H, W, n_class)
    offset = np.arange(label.size) * n_class + label.ravel()
    hot.ravel()[offset] = 1
    return hot.transpose((2, 0, 1))                 # (n_class, H, W)

class FCNDataset(Dataset):
    def __init__(self, root_path: str, n_class: int):
        super(FCNDataset, self).__init__()

        self.data_path = os.path.join(root_path, 'data')
        self.label_path = os.path.join(root_path, 'label')
        self.n_class = n_class
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.len = len(os.listdir(self.data_path))

    def __getitem__(self, index) -> T_co:
        image_name = os.listdir(self.data_path)[index]
        image_path = os.path.join(self.data_path, image_name)
        image = cv2.imread(image_path)          # BGR, (H, W, C)
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)                         # RGB, (H, W, C)
        image = self.trans(image)                                                   # RGB, (C, H, W)

        label_name = os.listdir(self.label_path)[index]
        label_path = os.path.join(self.label_path, label_name)
        label = np.load(label_path).astype(np.int8)                             # (H, W)
        label = one_hot(label, n_class=self.n_class)            # (n_class, H, W)
        label = torch.Tensor(label)

        return image, label

    def __len__(self):
        return self.len