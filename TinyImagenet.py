"""
Author: Meng Lee, mnicnc404
Date: 2018/06/04
References:
    - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
"""
import numpy as np
import os
import glob2 as glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision
import torch
EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


class TinyImageNet(Dataset):

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        # print(self.image_paths)
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}
        self.path_to_bbox = {}
        if self.split == 'train':
            for label in self.label_text_to_number:
                annfile = os.path.join(self.split_dir,label,'{}_boxes.txt'.format(label))
                with open(annfile, 'r') as fp:
                    for text in fp.readlines():
                        text = text.strip().split('\t')
                        imgname = os.path.join(self.split_dir,label,'images',text[0])
                        self.path_to_bbox[imgname] = (int(text[1])//2,int(text[2])//2,int(text[3])//2,int(text[4])//2)
        # print(self.path_to_bbox)
        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        # img = Image.open(file_path)
        img = cv2.imread(file_path)
        img =  torchvision.transforms.functional.to_pil_image(img)
        img = self.transform(img)
        if self.split == 'train':
            bbox = torch.zeros((3,32,32))
            x1,y1,x2,y2 = self.path_to_bbox[file_path]
            for i in range(32):
                for j in range(32):
                    if y1<=i and i<=y2 and x1<=j and j<=x2:
                        bbox[:,i,j] = 1.
                    else:
                        bbox[:,i,j] = -1.
            return img, self.labels[os.path.basename(file_path)], bbox
        else:
            return img, self.labels[os.path.basename(file_path)]


class Flower(Dataset):
    def __init__(self, path, split, transform=None):
        self.split = split
        self.transform = transform
        if split == 'train':
            loaded = np.load(os.path.join(path, 'masks.npy'))
            loaded = loaded.item()
            self.images = np.rollaxis((np.asarray(loaded['images'])), 3, 1)
            self.masks = np.rollaxis((np.asarray(loaded['masks'])), 3, 1)
            self.labels = np.asarray(loaded['labels']).astype(np.int64)
        else:
            loaded = np.load(os.path.join(path,'test.npz'))
            self.images = np.rollaxis((loaded['images']), 3, 1)/np.float32(255.)
            self.labels = loaded['labels'].astype(np.int64)

    def __getitem__(self,i):
        img = self.images[i]
        target = self.labels[i]

        if self.split == 'train':
            mask = self.masks[i]*255.
            mask = mask.transpose(1,2,0)
            mask = Image.fromarray(mask.astype('uint8'), 'RGB')
            mask = torchvision.transforms.functional.to_grayscale(mask)
            mask = torchvision.transforms.functional.to_tensor(mask)
            mask = (mask<0.2).float()
            return img,target, 2*mask-1
        else:
            return img, target

    def __len__(self):
        return len(self.labels)
