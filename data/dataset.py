import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pycocotools.coco import COCO
import h5py


class COCODataset(Dataset):
    def __init__(self, detection_path, annotation_file, transform=None, max_detections=50):
        self.detection_path = detection_path
        self.annotation_file = annotation_file
        self.transform = transform
        self.max_detections = max_detections
        super(COCODataset, self).__init__()

        # f = h5py.File(self.detections_path, 'r')

        id_map = COCO(annotation_file)

        self.data = []
        self.targets = []

        for id in list(id_map.anns.keys()):
            # img_id = id_map.anns[id]['image_id']
            # self.data.append(f[f'{img_id}_features'][()])
            self.data.append(id_map.anns[id]['image_id'])
            self.targets.append(id_map.anns[id]['caption'])

    def get_ref_dict(self):
        ref = {}
        for id, caption in zip(self.data, self.targets):
            ref[f'{id}'] = caption

        return ref

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f = h5py.File(self.detection_path, 'r')

        feature_id = self.data[idx]
        feature = f[f'{feature_id}_features'][()]
        caption = self.targets[idx]

        if feature.shape[0] < self.max_detections:
            padding = np.zeros((self.max_detections - feature.shape[0], feature.shape[1]))
            feature = np.concatenate([feature, padding])
        else:
            feature = feature[:self.max_detections]

        if self.transform != None:
            feature = self.transform(feature)

        feature = torch.squeeze(feature)

        f.close()

        return feature, caption, feature_id
