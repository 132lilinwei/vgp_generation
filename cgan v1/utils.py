import torch

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

# Others
def convert_to_device(device, *params):
    res = []
    for p in params:
        p = p.to(device)
        res.append(p)
    return res