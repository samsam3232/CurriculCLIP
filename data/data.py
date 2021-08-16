import logging
import os.path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

from model.clip import tokenize


class ConceptualCaption(Dataset):
    def __init__(self, input_filename, transforms, ims_root, sorted=False):
        logging.debug(f'Loading csv data from {input_filename}.')

        df = pd.read_csv(input_filename, sep='\t')

        if sorted:
            df = self.sort_csv(df)

        self.ims_root = ims_root
        self.images = df['filepath'].tolist()
        self.captions = df['title'].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def sort_csv(self, df):
        df['word_count'] = df['title'].str.replace(' \.', '.').str.split(' ').str.len()
        df = df.sort_values('word_count', ascending=False)
        return df

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        not_found = True
        while not_found:
            try:
                images = self.transforms(Image.open(os.path.join(self.ims_root, str(self.images[idx]))))
                texts = tokenize([str(self.captions[idx])])[0]
                not_found = False
            except:
                idx += 1
        return images, texts


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_imagenet_loader(args, preprocess_fns):

    preprocess_train, preprocess_val = preprocess_fns
    data_path = args.imagenet_val
    preprocess_fn = preprocess_val
    assert data_path

    dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    return dataloader


def get_cc_loader(args, preprocess_fn, is_train):

    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = ConceptualCaption(input_filename, preprocess_fn, ims_root=args.cc_root, sorted=args.sorted)
    num_samples = len(dataset)
    shuffle = is_train and not args.sorted

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                            pin_memory=True, drop_last=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_cc_loader(args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_cc_loader(args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet_loader(args, preprocess_fns)

    return data