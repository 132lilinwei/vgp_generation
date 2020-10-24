import os, sys
from os import listdir
from os.path import isfile, join
sys.path.append("..")

import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.data 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.models as models
from torchvision import transforms, utils

from PIL import Image
from sklearn.model_selection import train_test_split
from Flickr30kEntities.flickr30k_entities_utils import get_sentence_data, get_annotations 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
import time




            
def read_captions(path):
    all_names = []
    all_captions = []
    all_sentids = []
    with open(path, "r") as f:
        for line in f:
            name, sentid, raw_caption = line.strip().split(' ')
            caption = [int(item) for item in raw_caption.split(',')]
            all_names.append(name)
            all_sentids.append(int(sentid))
            all_captions.append(caption)
    return all_names, all_sentids, all_captions



class SupervisedDataset(Dataset):

    def __init__(self, config, feature_folder, caption_path, transform=None):
        self.config = config
        self.feature_folder = feature_folder
        self.caption_path = caption_path
        self.transform = transform
        self.caption_names, self.sentids, self.captions = read_captions(caption_path)
        
        # here we add START and END
        for i in range(len(self.captions)):
            self.captions[i] = [int(self.config.start_idx)] + self.captions[i] + [int(self.config.end_idx)]

            
    def __len__(self):
        return len(self.caption_names)

    def __getitem__(self, idx):
        filename = self.caption_names[idx]
        caption = self.captions[idx]
        image_feature = np.load(os.path.join(self.feature_folder, filename)+".npy")
        sample = {'image': torch.tensor(image_feature), "caption": torch.tensor(caption)}
        return sample
    
def supervised_collate_fn(batch, config):
    """
    This is the helper function for dataloader's collate_fn
    :param batch: dict {'image': list; 'caption':list}
        image: a list contains the image tensor 
        caption: a list contains the caption tensor, padded with START and END
    :return: final_batch: dict {'image': tensor, 'caption':tensor, 'length': tensor} sorted by caption length
    """
    sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    image_list = [item['image'] for item in sorted_batch]
    final_images = torch.stack(image_list)
    
    caption_list = [item['caption'] for item in sorted_batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    max_length = len(caption_list[0])
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)
    
    final_batch = {}
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    return final_batch


class DiscriminatorDataset(Dataset):

    def __init__(self, config, feature_folder, caption_path, transform=None):
        self.config = config
        self.feature_folder = feature_folder
        self.caption_path = caption_path
        self.transform = transform  
        self.caption_names, self.sentids, self.captions = read_captions(caption_path)
        
        # here we add START and END
        for i in range(len(self.captions)):
            self.captions[i] = [int(self.config.start_idx)] + self.captions[i] + [int(self.config.end_idx)]

            
    def __len__(self):
        return len(self.caption_names)

    def __getitem__(self, idx):
        filename = self.caption_names[idx]
        caption = self.captions[idx]
        image_feature = np.load(os.path.join(self.feature_folder, filename)+".npy")
        
        
        while True:
            rand = np.random.randint(0, len(self.caption_names))
            if self.caption_names[rand] == filename:
                continue
            wrong_caption = self.captions[rand]
            break
        
        sample = {'image': torch.tensor(image_feature), "caption": torch.tensor(caption), "wrong_caption":torch.tensor(wrong_caption)}
        return sample
    
def discriminator_collate_fn(batch, config):
    """
    This is the helper function for dataloader's collate_fn
    :param batch: dict {'image': list; 'caption':list}
        image: a list contains the image tensor 
        caption: a list contains the caption tensor, padded with START and END
    :return: final_batch: dict {'image': tensor; 
                                'caption':tensor; 
                                'length': tensor;
                                'wrong_caption': tensor; 
                                'wrong_length':tensor; } sorted by caption length (not by wrong caption length)
    """
    sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    image_list = [item['image'] for item in sorted_batch]
    final_images = torch.stack(image_list)
    
    caption_list = [item['caption'] for item in sorted_batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    max_length = len(caption_list[0])
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)
    
    wrong_caption_list = [item['wrong_caption'] for item in sorted_batch]
    wrong_caption_length = torch.tensor([len(item) for item in wrong_caption_list]).view(-1, 1)
    wrong_max_length = wrong_caption_length.max().item()
    final_wrong_captions = pad_sequence(wrong_caption_list, batch_first=True, padding_value=config.pad_idx)
    
    final_batch = {}
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    final_batch['wrong_caption'] = final_wrong_captions
    final_batch['wrong_length'] = wrong_caption_length
    return final_batch








if __name__ == '__main__':
    test_data = SupervisedDataset(config, feature_folder, caption_path)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, collate_fn=lambda batch: supervised_collate_fn(batch, config))
    discriminator_data = DiscriminatorDataset(config, feature_folder, caption_path)
    discriminator_loader = DataLoader(discriminator_data, batch_size=256, shuffle=False, collate_fn=lambda batch: discriminator_collate_fn(batch, config))
    for ct, data in enumerate(discriminator_loader):
        images = data["image"]
        captions = data["caption"]
        lengths = data['length']
        wrong_captions = data['wrong_caption']
        wrong_lengths = data['wrong_length']
        print(wrong_lengths)
        break






