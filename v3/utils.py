import torch
import json

# Read Embeddings and dictionary
def read_new_glove(dictionary_file, new_glove_file):
    print('reading new glove files:', new_glove_file)

    word_embed = []
    with open(new_glove_file, 'r') as f:
        for idx, line in enumerate(f):
            line_list = line.split()
            embed = [float(num) for num in line_list]
            word_embed.append(embed)

    for i in range(3):
        word_embed.append([0] * 300)    # add embeddings for <pad>, <\s>, <\e>

    w2i, i2w, pad_idx, start_idx, end_idx, vocab_size = read_dict(dictionary_file)

    return word_embed, w2i, i2w, pad_idx, start_idx, end_idx, vocab_size

def read_dict(dictionary_file):
    w2i = {}
    i2w = {}
    vocab_size = 0
    with open(dictionary_file, 'r') as f:
        for line in f:
            word, idx = line.strip().split(' ')
            w2i[word] = int(idx)
            i2w[int(idx)] = word
            vocab_size += 1

    # special case: <pad>, <\s>, <\e>
    pad_idx = vocab_size
    w2i['<pad>'] = vocab_size
    i2w[vocab_size] = '<pad>'
    vocab_size += 1

    start_idx = vocab_size
    w2i['<\s>'] = vocab_size
    i2w[vocab_size] = '<\s>'
    vocab_size += 1

    end_idx = vocab_size
    w2i['<\e>'] = vocab_size
    i2w[vocab_size] = '<\e>'
    vocab_size += 1

    return w2i, i2w, pad_idx, start_idx, end_idx, vocab_size

# -------------------
# IO to file
# -------------------
def write_to_files(sents, filepath):
    with open(filepath, 'w') as f:
        for sent in sents:
            f.write(sent + '\n')
            
def write_to_json_file(sent_list, target_list, caption_name_list, sentid_list, filepath):
    """
    Each caption maps to one dict: {'caption_name': str, 'pred': [], 'target': ['sent': str, 'sent_id': str]}
    """
    final_dict = {}
    for sent, target, caption_name, sentid in zip(sent_list, target_list, caption_name_list, sentid_list):
        temp_dict = final_dict.get(caption_name, {'caption_name': caption_name,
                                                 'pred': [], 'target': []})
        
        if sent not in temp_dict['pred']:
            temp_dict['pred'].append(sent)
#         temp_dict['pred'] = list(set(temp_dict['pred']))
        temp_dict['target'].append({'sent': target, 'sent_id': sentid})
        final_dict[caption_name] = temp_dict
    with open(filepath, 'w') as f:
        for sent, target, caption_name, sentid in zip(sent_list, target_list, caption_name_list, sentid_list):
            cur_dict = {'pred': sent, 'target': target, 'caption_name': caption_name, 'sentid': sentid}
            f.write(json.dumps(cur_dict) + '\n')
    return final_dict
            
def write_entity_to_json_file(entity_list, target_list, caption_name_list, entityid_list, desc_id_list, filepath):
    """
    Each entityid maps to one dict {'caption_name': , 'entityid':, 'pred': [], 'target': [{'entity':, 'desc_id': }]}
    """
    final_dict = {}
    for entity, target, caption, entityid, desc in zip(entity_list, target_list, caption_name_list, entityid_list, desc_id_list):
        temp_dict = final_dict.get(entityid, {'caption_name': caption,
                                             'entityid': entityid,
                                             'pred': [], 'target': []})
        if entity not in temp_dict['pred']:
            temp_dict['pred'].append(entity)
#         temp_dict['pred'] = list(set(temp_dict['pred']))
        temp_dict['target'].append({'entity': target, 'desc_id': desc})
        final_dict[entityid] = temp_dict
        
    with open(filepath, 'w') as f:
        for entityid in final_dict:
            f.write(json.dumps(final_dict[entityid]) + '\n')
    return final_dict

# -------------------
# Decode
# -------------------
def decode_sentence(pred_sents, i2w):
    """
    decode idx-sentences into string
    :param pred_sents: List[List[]], each element(list) is generated sentence with token idx
    :param i2w: dict, key: idx -> value : token str
    :return: sents_list: List[ str ], each element is generated sentence
    """
    sents_list = []
    for sents in pred_sents:
        sents_str = [i2w[token_idx] for token_idx in sents]
        sentence = ' '.join(sents_str)
        sents_list.append(sentence)
    return sents_list


# ------------------
# print config
# ------------------

def print_config(config):
    attrs = vars(config)
    print("Printing Config File:", flush = True)
    print('\n'.join("%s: %s" % item for item in attrs.items()), flush = True)

# ------------------
# Others
# ------------------
def convert_to_device(device, *params):
    res = []
    for p in params:
        if type(p) == list:
            p = [t.to(device) for t in p]
        else:
            p = p.to(device)
        res.append(p)
    return res

def compute_tf_idf(nums, idf_dict):
    """
    :param nums: a list of word_idx or a tensor to represent one phrase/description
    :param idf_dict: one dict: key is word-idx(str), value is idf value
    :return: return the tf-idf score of this phrase. Need to average?
    """
    score = 0
    if type(nums) == list:
        for num in nums:
            score += idf_dict.get(str(num), 0)
    else:
        for num in nums:
            score += idf_dict.get(str(num.item()), 0)
    return score

