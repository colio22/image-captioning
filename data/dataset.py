import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pycocotools.coco import COCO
import h5py


class COCODataset(Dataset):
    def __init__(self, detection_path, annotation_file, transform=None):
        self.detection_path = detection_path
        self.annotation_file = annotation_file
        self.transform = transform
        super(COCODataset, self).__init__()

        f = h5py.File(self.detections_path, 'r')

        id_map = COCO(annotation_file)

        self.data = []
        self.targets = []

        for id in list(id_map.anns.keys()):
            img_id = id_map.anns[id]['image_id']
            self.data.append(f[f'{img_id}_features'][()])
            self.targets.append(id_map.anns[id]['caption'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]
        caption = self.targets[idx]

        if self.transform != None:
            feature = self.transform(feature)

        return feature, caption
