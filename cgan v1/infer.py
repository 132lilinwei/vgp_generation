import utils

def infer(config, g_model, dataloader, i2w):
    pred_sents = []
    target_sents_list = []
    for idx, data in enumerate(dataloader):
        img_feature, target_sent, target_sent_len = data["image"], data["caption"], data["length"]
        img_feature, target_sent, target_sent_len = utils.convert_to_device(config.device, img_feature,
                                                                            target_sent, target_sent_len)

        if config.sample_method == 'greedy':
            # greedy search
            pred = g_model.greedy_generate(img_feature)
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
            
        # For train_loader, only predict first 10 batches.
#         if idx == 10:
#             break

    str_sent = utils.decode_sentence(pred_sents, i2w)   # List[ str ]
    target_sents_list = utils.decode_sentence(target_sents_list, i2w)
    return str_sent, target_sents_list