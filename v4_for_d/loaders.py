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
import json


class EntityAttnSupervisedDataset(Dataset):

    def __init__(self, config, image_feature_folder, entity_feature_folder, entity_caption_path, transform=None):
        self.config = config
        self.image_feature_folder = image_feature_folder
        self.entity_feature_folder = entity_feature_folder
        self.entity_caption_path = entity_caption_path
        self.transform = transform
        
        
        self.caption_names = [] # filename
        self.entity_ids = []
        self.desc_ids = []
        self.captions = []
        self.neighbors = []
        
        with open(entity_caption_path) as f:
            dataset = json.load(f)
        for file_item in dataset:
            filename = file_item['filename']
            
            neighbor_entities = []
            for entity_item in file_item['entities']:
                entity_id = entity_item['phrase_id']
                neighbor_entities.append(entity_id)
            
            
            for entity_item in file_item['entities']:
                entity_id = entity_item['phrase_id']
                for desc_item in entity_item['entity_descs']:
                    # if desc_item['tf_idf_level'] != 0:
                    #     continue
                    entity_desc_id = desc_item['entity_desc_id']
                    raw = desc_item['entity_desc'] # actually not raw, it is after processed
                    desc = desc_item['indexed']
                    # here we add START and END
                    desc = [int(self.config.start_idx)] + desc + [int(self.config.end_idx)]
                    
                    self.caption_names.append(filename)
                    self.entity_ids.append(entity_id)
                    self.desc_ids.append(entity_desc_id)
                    self.captions.append(desc)
                    self.neighbors.append(neighbor_entities)

            
    def __len__(self):
        return len(self.caption_names)

    def __getitem__(self, idx):
        filename = self.caption_names[idx]
        entity_id = self.entity_ids[idx]
        desc_id = self.desc_ids[idx]
        caption = self.captions[idx]
        neighbors = self.neighbors[idx]
        entity_feature = np.load(os.path.join(self.entity_feature_folder, str(entity_id)+".npy"))
        
        
        neighbor_feature = []
        for neighbor_id in neighbors:
            temp = np.load(os.path.join(self.entity_feature_folder, str(neighbor_id)+".npy"))
            neighbor_feature.append(temp)
        neighbor_feature = np.stack(neighbor_feature)
        image_feature = np.load(os.path.join(self.image_feature_folder, str(filename)+".npy"))
        
            
        sample = {'entity_feature': torch.tensor(entity_feature), "caption": torch.tensor(caption), \
                    "caption_name": filename, "entity_id": entity_id, 'desc_id':desc_id, \
                    "neighbor_feature": torch.tensor(neighbor_feature), 'image_feature': torch.tensor(image_feature)}
        
        return sample
    
def entity_attn_supervised_collate_fn(batch, config):
    """
    This is the helper function for dataloader's collate_fn
    :param batch: dict {'image': list; 'caption':list}
        image: a list contains the image tensor 
        caption: a list contains the caption tensor, padded with START and END
    :return: final_batch: dict {'image': tensor, 'caption':tensor, 'length': tensor} sorted by caption length
    """
    # sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    entity_list = [item['entity_feature'] for item in batch]
    final_entities = torch.stack(entity_list)
    
    caption_list = [item['caption'] for item in batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)
    
    neighbor_list = [item['neighbor_feature'] for item in batch]
    
    image_list = [item['image_feature'] for item in batch]
    final_images = torch.stack(image_list)

    caption_name_list = [item['caption_name'] for item in batch]
    entity_id_list = [item['entity_id'] for item in batch]
    desc_id_list = [item['desc_id'] for item in batch]
    
    final_batch = {}
    final_batch['entity'] = final_entities
    final_batch['neighbor'] = neighbor_list
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    final_batch['caption_name'] = caption_name_list
    final_batch['entity_id'] = entity_id_list
    final_batch['desc_id'] = desc_id_list
    return final_batch

class EntityAttnDiscriminatorDataset(Dataset):

    def __init__(self, config, image_feature_folder, entity_feature_folder, entity_caption_path, transform=None):
        self.config = config
        self.image_feature_folder = image_feature_folder
        self.entity_feature_folder = entity_feature_folder
        self.entity_caption_path = entity_caption_path
        self.transform = transform
        
        
        
        self.caption_names = [] # filename
        self.entity_ids = []
        self.desc_ids = []
        self.captions = []
        self.neighbors = []
    
        
        with open(entity_caption_path) as f:
            dataset = json.load(f)
        for file_item in dataset:
            filename = file_item['filename']
            
            neighbor_entities = []
            for entity_item in file_item['entities']:
                entity_id = entity_item['phrase_id']
                neighbor_entities.append(entity_id)
                
            for entity_item in file_item['entities']:
                entity_id = entity_item['phrase_id']
                for desc_item in entity_item['entity_descs']:
                    # if desc_item['tf_idf_level'] != 0:
                    #     continue
                    entity_desc_id = desc_item['entity_desc_id']
                    raw = desc_item['entity_desc'] # actually not raw, it is after processed
                    desc = desc_item['indexed']
                    # here we add START and END
                    desc = [int(self.config.start_idx)] + desc + [int(self.config.end_idx)]
                    
                    self.caption_names.append(filename)
                    self.entity_ids.append(entity_id)
                    self.desc_ids.append(entity_desc_id)
                    self.captions.append(desc)
                    self.neighbors.append(neighbor_entities)

            
    def __len__(self):
        return len(self.caption_names)

    def __getitem__(self, idx):
        filename = self.caption_names[idx]
        entity_id = self.entity_ids[idx]
        desc_id = self.desc_ids[idx]
        caption = self.captions[idx]
        neighbors = self.neighbors[idx]
        entity_feature = np.load(os.path.join(self.entity_feature_folder, str(entity_id)+".npy"))
        
        neighbor_feature = []
        for neighbor_id in neighbors:
            temp = np.load(os.path.join(self.entity_feature_folder, str(neighbor_id)+".npy"))
            neighbor_feature.append(temp)
        neighbor_feature = np.stack(neighbor_feature)
        image_feature = np.load(os.path.join(self.image_feature_folder, str(filename)+".npy"))
        
        
        while True:
            rand = np.random.randint(0, len(self.caption_names))
            if self.entity_ids[rand] == entity_id:
                continue
            wrong_caption = self.captions[rand]
            break
    
        
        sample = {'entity_feature': torch.tensor(entity_feature), "caption": torch.tensor(caption), "wrong_caption":torch.tensor(wrong_caption), \
                    "caption_name": filename, "entity_id": entity_id, "desc_id": desc_id, \
                    "wrong_caption_name": self.caption_names[rand], "wrong_entity_id": self.entity_ids[rand], "wrong_desc_id":self.desc_ids[rand], \
                    "neighbor_feature": torch.tensor(neighbor_feature), 'image_feature': torch.tensor(image_feature)}
        return sample
    
def entity_attn_discriminator_collate_fn(batch, config):
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
    # sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    entity_list = [item['entity_feature'] for item in batch]
    final_entities = torch.stack(entity_list)
    
    neighbor_list = [item['neighbor_feature'] for item in batch]
    
    image_list = [item['image_feature'] for item in batch]
    final_images = torch.stack(image_list)
    
    caption_list = [item['caption'] for item in batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)
    
    wrong_caption_list = [item['wrong_caption'] for item in batch]
    wrong_caption_length = torch.tensor([len(item) for item in wrong_caption_list]).view(-1, 1)
    final_wrong_captions = pad_sequence(wrong_caption_list, batch_first=True, padding_value=config.pad_idx)

    caption_name_list = [item['caption_name'] for item in batch]
    entity_id_list = [item['entity_id'] for item in batch]
    desc_id_list = [item['desc_id'] for item in batch]
    wrong_caption_name_list = [item['wrong_caption_name'] for item in batch]
    wrong_entity_id_list = [item['wrong_entity_id'] for item in batch]
    wrong_desc_id_list = [item['wrong_desc_id'] for item in batch]
    
    final_batch = {}
    final_batch['entity'] = final_entities
    final_batch['neighbor'] = neighbor_list
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    final_batch['wrong_caption'] = final_wrong_captions
    final_batch['wrong_length'] = wrong_caption_length
    final_batch['caption_name'] = caption_name_list
    final_batch['entity_id'] = entity_id_list
    final_batch['desc_id'] = desc_id_list
    final_batch['wrong_caption_name'] = wrong_caption_name_list
    final_batch['wrong_entity_id'] = wrong_entity_id_list
    final_batch['wrong_desc_id'] = wrong_desc_id_list
    return final_batch






class EntitySupervisedDataset(Dataset):

    def __init__(self, config, feature_folder, entity_caption_path, transform=None):
        self.config = config
        self.feature_folder = feature_folder
        self.entity_caption_path = entity_caption_path
        self.transform = transform
        
        
        self.caption_names = [] # filename
        self.entity_ids = []
        self.desc_ids = []
        self.captions = []
        
        with open(entity_caption_path) as f:
            dataset = json.load(f)
        for file_item in dataset:
            filename = file_item['filename']
            for entity_item in file_item['entities']:
                entity_id = entity_item['phrase_id']
                for desc_item in entity_item['entity_descs']:
                    entity_desc_id = desc_item['entity_desc_id']
                    raw = desc_item['entity_desc'] # actually not raw, it is after processed
                    desc = desc_item['indexed']
                    # here we add START and END
                    desc = [int(self.config.start_idx)] + desc + [int(self.config.end_idx)]
                    
                    self.caption_names.append(filename)
                    self.entity_ids.append(entity_id)
                    self.desc_ids.append(entity_desc_id)
                    self.captions.append(desc)

            
    def __len__(self):
        return len(self.caption_names)

    def __getitem__(self, idx):
        filename = self.caption_names[idx]
        entity_id = self.entity_ids[idx]
        desc_id = self.desc_ids[idx]
        caption = self.captions[idx]
        image_feature = np.load(os.path.join(self.feature_folder, str(entity_id)+".npy"))
        sample = {'image': torch.tensor(image_feature), "caption": torch.tensor(caption), \
                    "caption_name": filename, "entity_id": entity_id, 'desc_id':desc_id}
        return sample
    
def entity_supervised_collate_fn(batch, config):
    """
    This is the helper function for dataloader's collate_fn
    :param batch: dict {'image': list; 'caption':list}
        image: a list contains the image tensor 
        caption: a list contains the caption tensor, padded with START and END
    :return: final_batch: dict {'image': tensor, 'caption':tensor, 'length': tensor} sorted by caption length
    """
    # sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    image_list = [item['image'] for item in batch]
    final_images = torch.stack(image_list)
    
    caption_list = [item['caption'] for item in batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)

    caption_name_list = [item['caption_name'] for item in batch]
    entity_id_list = [item['entity_id'] for item in batch]
    desc_id_list = [item['desc_id'] for item in batch]
    
    final_batch = {}
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    final_batch['caption_name'] = caption_name_list
    final_batch['entity_id'] = entity_id_list
    final_batch['desc_id'] = desc_id_list
    return final_batch

class EntityDiscriminatorDataset(Dataset):

    def __init__(self, config, feature_folder, entity_caption_path, transform=None):
        self.config = config
        self.feature_folder = feature_folder
        self.entity_caption_path = entity_caption_path
        self.transform = transform
        
        
        self.caption_names = [] # filename
        self.entity_ids = []
        self.desc_ids = []
        self.captions = []
    
        
        with open(entity_caption_path) as f:
            dataset = json.load(f)
        for file_item in dataset:
            filename = file_item['filename']
            for entity_item in file_item['entities']:
                entity_id = entity_item['phrase_id']
                for desc_item in entity_item['entity_descs']:
                    entity_desc_id = desc_item['entity_desc_id']
                    raw = desc_item['entity_desc'] # actually not raw, it is after processed
                    desc = desc_item['indexed']
                    # here we add START and END
                    desc = [int(self.config.start_idx)] + desc + [int(self.config.end_idx)]
                    
                    self.caption_names.append(filename)
                    self.entity_ids.append(entity_id)
                    self.desc_ids.append(entity_desc_id)
                    self.captions.append(desc)

            
    def __len__(self):
        return len(self.caption_names)

    def __getitem__(self, idx):
        filename = self.caption_names[idx]
        entity_id = self.entity_ids[idx]
        desc_id = self.desc_ids[idx]
        caption = self.captions[idx]
        image_feature = np.load(os.path.join(self.feature_folder, str(entity_id)+".npy"))
        
        
        while True:
            rand = np.random.randint(0, len(self.caption_names))
            if self.entity_ids[rand] == entity_id:
                continue
            wrong_caption = self.captions[rand]
            break
        sample = {'image': torch.tensor(image_feature), "caption": torch.tensor(caption), "wrong_caption":torch.tensor(wrong_caption), \
                    "caption_name": filename, "entity_id": entity_id, "desc_id": desc_id, \
                    "wrong_caption_name": self.caption_names[rand], "wrong_entity_id": self.entity_ids[rand], "wrong_desc_id":self.desc_ids[rand]}
        return sample
    
def entity_discriminator_collate_fn(batch, config):
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
    # sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    image_list = [item['image'] for item in batch]
    final_images = torch.stack(image_list)
    
    caption_list = [item['caption'] for item in batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)
    
    wrong_caption_list = [item['wrong_caption'] for item in batch]
    wrong_caption_length = torch.tensor([len(item) for item in wrong_caption_list]).view(-1, 1)
    final_wrong_captions = pad_sequence(wrong_caption_list, batch_first=True, padding_value=config.pad_idx)

    caption_name_list = [item['caption_name'] for item in batch]
    entity_id_list = [item['entity_id'] for item in batch]
    desc_id_list = [item['desc_id'] for item in batch]
    wrong_caption_name_list = [item['wrong_caption_name'] for item in batch]
    wrong_entity_id_list = [item['wrong_entity_id'] for item in batch]
    wrong_desc_id_list = [item['wrong_desc_id'] for item in batch]
    
    final_batch = {}
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    final_batch['wrong_caption'] = final_wrong_captions
    final_batch['wrong_length'] = wrong_caption_length
    final_batch['caption_name'] = caption_name_list
    final_batch['entity_id'] = entity_id_list
    final_batch['desc_id'] = desc_id_list
    final_batch['wrong_caption_name'] = wrong_caption_name_list
    final_batch['wrong_entity_id'] = wrong_entity_id_list
    final_batch['wrong_desc_id'] = wrong_desc_id_list
    return final_batch

            
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
        sample = {'image': torch.tensor(image_feature), "caption": torch.tensor(caption), \
                    "caption_name": filename, "sentid": self.sentids[idx]}
        return sample
    
def supervised_collate_fn(batch, config):
    """
    This is the helper function for dataloader's collate_fn
    :param batch: dict {'image': list; 'caption':list}
        image: a list contains the image tensor 
        caption: a list contains the caption tensor, padded with START and END
    :return: final_batch: dict {'image': tensor, 'caption':tensor, 'length': tensor} sorted by caption length
    """
    # sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    image_list = [item['image'] for item in batch]
    final_images = torch.stack(image_list)
    
    caption_list = [item['caption'] for item in batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)

    caption_name_list = [item['caption_name'] for item in batch]
    sentid_list = [item['sentid'] for item in batch]
    
    final_batch = {}
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    final_batch['caption_name'] = caption_name_list
    final_batch['sentid'] = sentid_list
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
        
        sample = {'image': torch.tensor(image_feature), "caption": torch.tensor(caption), "wrong_caption":torch.tensor(wrong_caption), \
                    "caption_name": filename, "sentid": self.sentids[idx], "wrong_sentid": self.sentids[rand]}
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
    # sorted_batch = sorted(batch, key=lambda x: x['caption'].shape[0], reverse=True)
    image_list = [item['image'] for item in batch]
    final_images = torch.stack(image_list)
    
    caption_list = [item['caption'] for item in batch]
    caption_length = torch.tensor([len(item) for item in caption_list]).view(-1, 1)
    final_captions = pad_sequence(caption_list, batch_first=True, padding_value=config.pad_idx)
    
    wrong_caption_list = [item['wrong_caption'] for item in batch]
    wrong_caption_length = torch.tensor([len(item) for item in wrong_caption_list]).view(-1, 1)
    final_wrong_captions = pad_sequence(wrong_caption_list, batch_first=True, padding_value=config.pad_idx)

    caption_name_list = [item['caption_name'] for item in batch]
    sentid_list = [item['sentid'] for item in batch]
    wrong_sentid_list = [item['wrong_sentid'] for item in batch]
    
    final_batch = {}
    final_batch['image'] = final_images
    final_batch['caption'] = final_captions
    final_batch['length'] = caption_length
    final_batch['wrong_caption'] = final_wrong_captions
    final_batch['wrong_length'] = wrong_caption_length
    final_batch['caption_name'] = caption_name_list
    final_batch['sentid'] = sentid_list
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






