import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm
from PIL import Image
from torchvision.transforms.functional import to_tensor


def read_image_names(path, step):
    assert step in ['train', 'valid', 'test']
    path = os.path.join(path, step, 'image')  #if step in ['train', 'valid'] else os.path.join(path, step, 'image')
    names = []
    for idx, file in enumerate(tqdm(os.listdir(path))):
        name = os.path.splitext(file)[0]
        names.append(name)
        '''if idx > 1000:
            return names'''
    return names


class BuildingDataset(Dataset):
    def __init__(self, split, dataset_name ='WHU', counting=False):
        assert dataset_name in ['WHU', 'NewInria', 'CrowdAI', 'Mass','NZ32','CTC'], \
            'dataset_name must be in [WHU, NewInria, CrowdAI, Mass, NZ32, CTC]'
        self.suffix = 'jpg' if dataset_name == 'CrowdAI' else 'png'
        dataset_dict = {'WHU': r'C:\ZTB\Dataset\WHUBuilding',
                        'NewInria':r'C:\ZTB\Dataset\NewInria',
                        'Mass':r'E:\DataSet\Massachusetts',
                        'CrowdAI':r'C:\ZTB\Dataset\CrowdAI_split',
                        'NZ32':r'E:\DataSet\NZ32km2',
                        'CTC':r'C:\ZTB\Dataset\CTC'}
        self.path = dataset_dict[dataset_name]
        self.split = split
        self.names = read_image_names(self.path, self.split)
        self.counting=counting

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.split, 'image', f'{self.names[idx]}.{self.suffix}')
        label_path = os.path.join(self.path, self.split, 'label', f'{self.names[idx]}.{self.suffix}')
        edge_path = os.path.join(self.path, self.split, 'edge', f'{self.names[idx]}.{self.suffix}')
        label = Image.open(label_path)
        edge = Image.open(edge_path) if not self.split in ['test'] else None
        image = to_tensor(Image.open(image_path))
        tensor_label = to_tensor(label)  # .long()
        tensor_label[tensor_label > 0.5] = 1
        tensor_label[tensor_label < 1] = 0
        if edge is not None:
            tensor_edge = to_tensor(edge)
            tensor_edge[tensor_edge > 0.5] = 1
            tensor_edge[tensor_edge < 1] = 0
        else:
            tensor_edge = None
        if self.counting:
            buildings_centre = Image.open(os.path.join(self.path, self.step, 'density', f'{self.names[idx]}.{self.suffix}'))
            buildings_centre = to_tensor(buildings_centre)
            return image, buildings_centre, self.names[idx]
        else:
            if self.split == 'test':
                return image, tensor_label, self.names[idx]
            else:
                return image, tensor_label, tensor_edge

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    dataset = BuildingDataset(dataset_name='CrowdAI', split='test', counting=False)
    ratio = []
    for img, label, edge in dataset:
    #for img, density, count in dataset:
        img = img.permute(1, 2, 0)
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title('image')

        '''density = density.squeeze()
        plt.subplot(1, 4, 2)
        plt.imshow(density)
        plt.title('density')'''
        label = label.squeeze()
        plt.subplot(1, 4, 2)
        plt.imshow(label)
        plt.title('mask')

        '''edge = edge.squeeze()
        total_pxl = 256 * 256
        edge_pxl = torch.count_nonzero(edge)
        edge_ratio = edge_pxl / total_pxl
        ratio.append(edge_ratio.numpy())'''
        # print('ratio:',edge_pxl/total_pxl)
        '''plt.subplot(1, 4, 3)
        plt.imshow(edge)
        plt.title('edge')'''

        #dist = dist.squeeze()
        '''plt.subplot(1, 4, 4)
        plt.imshow(dist)
        plt.title('distance')'''
        plt.show()
    #print(np.max(ratio))
