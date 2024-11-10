import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pycocotools.coco import COCO
import h5py
import random
import sys


class COCODataset(Dataset):
    """
    Custom Dataset class composed of pre-extracted features from the COOC image set.
    The class stores the image IDs and captions and then reads the actual feature
    data only when necessary.
    """

    def __init__(self, detection_path, annotation_file, transform=None, max_detections=50, limit=0):
        self.detection_path = detection_path        # Path to pre-extracted feature HDF file
        self.annotation_file = annotation_file      # Path to directory of COCO annotations
        self.transform = transform                  # Transform to apply to dataset
        self.max_detections = max_detections        # Maximum number of features per image to consider
        super(COCODataset, self).__init__()

        # Get all annotation IDs
        id_map = COCO(annotation_file)

        # Lists to hold image IDs and corresponding captions
        self.data = []
        self.targets = []

        # Loop over all annaotation IDs
        for id in list(id_map.anns.keys()):
            # Save associated image ID and caption
            self.data.append(id_map.anns[id]['image_id'])
            self.targets.append(id_map.anns[id]['caption'])

        # If valid limit was given, restrict dataset to only that number of items
        if limit != 0 and limit < len(self.data):
            # Lists to hold reduced data
            slim_data = []
            slim_targets = []

            # Copy <limit> number of data items
            for i in range(limit):
                slim_data.append(self.data[i])
                slim_targets.append(self.targets[i])

            # Assign dataset to only be reduced set
            self.data = slim_data
            self.targets = slim_targets


    def get_ref_dict(self):
        """
        Returns a dictionary of all IDs and their corresponding captions
        """

        # Save each caption in a dictionary with associated id
        ref = {}
        for id, caption in zip(self.data, self.targets):
            ref[f'{id}'] = caption

        return ref

    def __len__(self):
        """
        Returns length of dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets an item from the dataset at idx
        """

        # Read from HDF5 file
        with h5py.File(self.detection_path, 'r') as f:
            # Loop until finding a valid key.
            # This failsafe measure was introduced because there are a handful
            # of image IDs that do not have an associated entry in the feature
            # detection file.
            valid_key = False
            while (valid_key == False) and (idx < len(self.data)):
                print(f'In loop for {idx}')
                feature_id = self.data[idx]
                try:
                    feature = f[f'{feature_id}_features'][()]
                    valid_key = True
                except KeyError:    # If error encountered, skip and move to next line until valid
                    print(f"Failed while accessing {feature_id}. Skipping to next feature in list.")
                    valid_key = False
                    idx += 1

                    # Wrap around
                    if idx >= len(self.data):
                        idx = 0

            # Save caption of idx
            caption = self.targets[idx]

        # Pad entry to 50 x 2048 shape
        if feature.shape[0] < self.max_detections:
            padding = np.zeros((self.max_detections - feature.shape[0], feature.shape[1]))
            feature = np.concatenate([feature, padding])
        else:
            feature = feature[:self.max_detections]

        # Apply transform if given
        if self.transform != None:
            feature = self.transform(feature)

        # Fix dimensionality
        feature = torch.squeeze(feature)

        return feature, caption, feature_id
