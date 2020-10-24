
"""
Preprocess glove embeddings
Combine our own dictionary with glove embeddings
"""
def read_old_glove(filepath):
    """
    get word2idx and word_embed
    NOTE: set <PAD> as word_idx = 0, embed_size is 300d
          Here all elements in word_embed is str rather than int/float
    """
    print('reading glove files:', filepath)

    word2idx = {}
    word_embed = [['0'] * 300]    # word_embed[0] = [0] * 300, represent the <PAD>

    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            line_list = line.split()
            word = ' '.join(line_list[: len(line_list)-300])
            embed = [num for num in line_list[len(line_list)-300:]]

            word2idx[word] = idx + 1
            word_embed.append(embed)

    return word2idx, word_embed

def generate_new_glove(glove_file, dictionary_file, new_glove_file):
    glove_w2i, glove_embed = read_old_glove(glove_file)

    # read dictionary
    w2i = []
    unk_count = 0
    with open(dictionary_file, 'r') as f:
        for line in f:
            word, idx = line.strip().split(' ')
            # if word is out-of-box, set it as [0] * 300 vector
            w2i.append(glove_embed[glove_w2i.get(word, 0)])
            if word not in glove_w2i:
                unk_count += 1
    print('unk_count: ', unk_count)

    with open(new_glove_file, 'w') as f:
        for embed in w2i:
            f.write(' '.join(embed) + '\n')

if __name__ == '__main__':
    # pass
    generate_new_glove('embeddings/glove.6B.300d.txt', 'data/train_dictionary.txt', 'new_glove_train.txt')
    # print('a')