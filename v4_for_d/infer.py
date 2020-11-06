import utils

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
            pred = g_model.greedy_generate(entity_feature, neighbor_feature_list, img_feature)
            pred_sents.extend(pred)
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
    
    g_model.eval()
    for idx, data in enumerate(dataloader):
        entity_feature, neighbor_feature_list, img_feature, target_entity, target_entity_len = data['entity'], data['neighbor'], data["image"], data["caption"], data["length"]
        entity_feature, neighbor_feature_list, img_feature, target_sent, target_sent_len = utils.convert_to_device(config.device, entity_feature, neighbor_feature_list, img_feature, target_entity, target_entity_len)
        
        if config.sample_method == 'greedy':
            # greedy search
            pred = g_model.greedy_generate(entity_feature, neighbor_feature_list, img_feature)
            pred_entity.extend(pred)
        else:
            # beam search
            pass
        
        # convert target_sent to List[List[]]
        for entity in target_entity:
            temp = []
            for idx in entity:
                idx = idx.item()
                if idx != config.start_idx and idx != config.end_idx and idx != config.pad_idx:
                    temp.append(idx)
            target_entity_list.append(temp)
            
        caption_name_list.extend(data['caption_name'])
        entityid_list.extend(data['entity_id'])
        desc_id_list.extend(data['desc_id'])
        
    str_entity = utils.decode_sentence(pred_entity, i2w)   # List[ str ]
    target_entity_list = utils.decode_sentence(target_entity_list, i2w)
    return str_entity, target_entity_list, caption_name_list, entityid_list, desc_id_list