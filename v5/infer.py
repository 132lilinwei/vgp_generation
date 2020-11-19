import utils
import numpy as np

def infer(config, g_model, dataloader, i2w):
    pred_sents = []
    target_sents_list = []
    caption_name_list = []
    sentid_list = []
    g_model.eval()
    for idx, data in enumerate(dataloader):
        entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len = data['entity'], data['neighbor'], data["image"], data["caption"], data["length"]
        entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len = utils.convert_to_device(config.device, entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len)

        if config.sample_method == 'greedy':
            # greedy search
            l1_pred, l2_pred = g_model.greedy_generate(entity_feature, neighbor_feature_list, img_feature)
            pred_sents.extend(l1_pred)
        else:
            # beam search
            pass

        # convert target_sent to List[List[]]
        for sent in target_sent:
            temp = []
            for idx in sent:
                idx = idx.item()
                if idx != config.start_idx and idx != config.end_idx and idx != config.pad_idx:
                    temp.append(idx)
            target_sents_list.append(temp)
            
        caption_name_list.extend(data['caption_name'])
        sentid_list.extend(data['sentid'])
        # For train_loader, only predict first 10 batches.
#         if idx == 10:
#             break

    str_sent = utils.decode_sentence(pred_sents, i2w)   # List[ str ]
    target_sents_list = utils.decode_sentence(target_sents_list, i2w)
    return str_sent, target_sents_list, caption_name_list, sentid_list

def entity_infer(config, g_model, dataloader, i2w):
    pred_entity = []
    target_entity_list = []
    caption_name_list = []
    entityid_list = []
    desc_id_list = []
    
    l2_pred_entity = []
    l2_target_entity_list = []
    l2_caption_name_list = []
    l2_entityid_list = []
    l2_desc_id_list = []
    
    g_model.eval()
    for idx, data in enumerate(dataloader):
        entity_feature, neighbor_feature_list, img_feature, target_entity, l2_target_entity, exist_mask = data['entity'], data['neighbor'], data["image"], data["caption"], data['fine_caption'], data['fine_exist']
        entity_feature, neighbor_feature_list, img_feature, target_sent, l2_target_entity = utils.convert_to_device(config.device, entity_feature, neighbor_feature_list, img_feature, target_entity, l2_target_entity)
        
        if config.sample_method == 'greedy':
            # greedy search
            l1_pred, l2_pred = g_model.greedy_generate(entity_feature, neighbor_feature_list, img_feature)
            pred_entity.extend(l1_pred)

        else:
            if idx % 10 == 0:
                print("Beam Search Progress:", idx,"/", len(dataloader))
            pred, prob = g_model.beam_sample(entity_feature, neighbor_feature_list, img_feature, beam_size = config.beam_size)
            pred_entity.extend(pred)
#             print(pred)
#             print(utils.decode_sentence(pred, i2w))
        
        # convert target_sent to List[List[]]
        for entity in target_entity:
            temp = []
            for idx in entity:
                idx = idx.item()
                if idx != config.start_idx and idx != config.end_idx and idx != config.pad_idx:
                    temp.append(idx)
            target_entity_list.append(temp)
        
        exist_mask = exist_mask.view(-1)
        for i, mask in enumerate(exist_mask):
            if mask == 1:
                l2_caption_name_list.append(data['caption_name'][i])
                l2_entityid_list.append(data['entity_id'][i])
                l2_desc_id_list.append(data['desc_id'][i])
                l2_pred_entity.append(l2_pred[i])
                
                temp = []
                entity = l2_target_entity[i]
                for idx in entity:
                    idx = idx.item()
                    if idx != config.start_idx and idx != config.end_idx and idx != config.pad_idx:
                        temp.append(idx)
                l2_target_entity_list.append(temp)
            
        caption_name_list.extend(data['caption_name'])
        entityid_list.extend(data['entity_id'])
        desc_id_list.extend(data['desc_id'])
        
    str_entity = utils.decode_sentence(pred_entity, i2w)   # List[ str ]
    l2_str_entity = utils.decode_sentence(l2_pred_entity, i2w)
    target_entity_list = utils.decode_sentence(target_entity_list, i2w)
    l2_target_entity_list = utils.decode_sentence(l2_target_entity_list, i2w)
    return str_entity, target_entity_list, caption_name_list, entityid_list, desc_id_list, l2_str_entity, l2_target_entity_list,  l2_caption_name_list, l2_entityid_list, l2_desc_id_list



def entity_diversity_infer(config, g_model, dataloader, i2w):
    pred_entity = [] # list of predicted captions
    target_entity_list = [] # List of target captions
    caption_name_list = []
    entityid_list = []
    desc_id_list = []
    
    l2_pred_entity = []
    l2_target_entity_list = []
    l2_desc_id_list = []

    
    g_model.eval()
    for idx, data in enumerate(dataloader):
        entity_feature, neighbor_feature_list, img_feature, target_entity, l2_target_entity, exist_mask = data['entity'], data['neighbor'], data["image"], data["caption"], data['fine_caption'], data['fine_exist']
        entity_feature, neighbor_feature_list, img_feature, target_sent, l2_target_entity = utils.convert_to_device(config.device, entity_feature, neighbor_feature_list, img_feature, target_entity, l2_target_entity)
        
        if config.sample_method == 'greedy':
            # greedy search
            l1_pred, l2_pred = g_model.greedy_generate(entity_feature, neighbor_feature_list, img_feature)
            pred_entity.extend(l1_pred)
            l2_pred_entity.extent(l2_pred)

        else:
            if idx % 10 == 0:
                print("Beam Search Progress:", idx,"/", len(dataloader))
            pred, prob = g_model.beam_sample(entity_feature, neighbor_feature_list, img_feature, beam_size = config.beam_size)
            pred_entity.extend(pred)
#             print(pred)
#             print(utils.decode_sentence(pred, i2w))
        
        # convert target_sent to List[List[]]
        for entity in target_entity:
            temp = []
            for idx in entity:
                idx = idx.item()
                if idx != config.start_idx and idx != config.end_idx and idx != config.pad_idx:
                    temp.append(idx)
            target_entity_list.append(temp)
            
        for entity in l2_target_entity:
            temp = []
            for idx in entity:
                idx = idx.item()
                if idx != config.start_idx and idx != config.end_idx and idx != config.pad_idx:
                    temp.append(idx)
            l2_target_entity_list.append(temp)
        
            
        caption_name_list.extend(data['caption_name'])
        entityid_list.extend(data['entity_id'])
        desc_id_list.extend(data['desc_id'])
        l2_desc_id_list.extend(data['fine_desc_id'])
        
    str_entity = utils.decode_sentence(pred_entity, i2w)   # List[ str ]
    l2_str_entity = utils.decode_sentence(l2_pred_entity, i2w)
    target_entity_list = utils.decode_sentence(target_entity_list, i2w)
    l2_target_entity_list = utils.decode_sentence(l2_target_entity_list, i2w)
    return str_entity, target_entity_list, caption_name_list, entityid_list, desc_id_list, l2_str_entity, l2_target_entity_list,  None, None, l2_desc_id_list