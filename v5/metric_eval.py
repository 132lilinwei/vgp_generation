from nlgeval import compute_metrics
from nlgeval import compute_individual_metrics
from nlgeval import NLGEval
import numpy as np
from cider.cider import Cider
import pickle

"""
need to install nlgeval package
Download url and use instruction: https://github.com/Maluuba/nlg-eval

Download and setup:
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
"""

def eval_txt(model_generated_file, target_file):
    """
    Input is txt file
    It will directly print out the result
    
    One example:
    Bleu_1: 0.263080
    Bleu_2: 0.135471
    Bleu_3: 0.074235
    Bleu_4: 0.043782
    METEOR: 0.096038
    ROUGE_L: 0.247746
    CIDEr: 0.238100
    """
    scores = compute_metrics(model_generated_file, [target_file], no_glove=True, no_skipthoughts=True)
    
def eval_one_sent(model_generated_sent, target_sent):
    """
    model_generated_sent is str, target_sent can be List[str] or str
    One example of scores:
    scores = {'Bleu_1': 0.6666666662222226, 'Bleu_2': 0.5773502687566137, 'Bleu_3': 6.933612736957944e-06, 'Bleu_4': 4.272870060579654e-06, 'METEOR': 0.32942735311884513, 'ROUGE_L': 0.6666666666666666, 'CIDEr': 0.0}
    """
    target_sent = target_sent if type(target_sent)==list else [target_sent]
    scores = compute_individual_metrics(hyp=model_generated_sent, ref=target_sent, no_glove=True, no_skipthoughts=True)
    return scores

def eval_sent_list(model_sent_list, target_sent_list):
    """
    Not to use, it is wrong!
    Input is List[str], List[str]/List[List[str]] 
    
    One example:
    {'Bleu_1': 0.8333333327777782, 'Bleu_2': 0.7886751340033071, 'Bleu_3': 0.5000034663341466, 'Bleu_4': 0.015813524723354844, 'METEOR': 0.6910678648089302, 'ROUGE_L': 0.8333333333333333, 'CIDEr': 0.0}
    """
    eval_model = NLGEval(no_glove=True, no_skipthoughts=True)
    res = {}
    for m_sent, t_sent in zip(model_sent_list, target_sent_list):
        t_sent = t_sent if type(t_sent) == list else [t_sent]
        temp_res = eval_model.compute_individual_metrics(hyp=m_sent, ref=t_sent)
        for key in temp_res:
            res[key] = res.get(key, 0) + temp_res[key]
    for key in res:
        res[key] = res[key] / len(model_sent_list)
    return res

def eval_entity(final_dict):
    model_generate = []
    target = [[] for i in range(5)]
    for entity_id in final_dict:
        model_generate.append(final_dict[entity_id]['pred'][0])
        t = [d['entity'] for d in final_dict[entity_id]['target']]
        for i in range(5):
            target[i].append(t[i] if i < len(t) else '')
    
    eval_model = NLGEval(no_glove=True, no_skipthoughts=True)
    res = eval_model.compute_metrics(target, model_generate)
    print(res)
    return res

def eval_caption(final_dict):
    model_generate = []
    target = [[] for i in range(5)]
    for caption_name in final_dict:
        model_generate.append(final_dict[caption_name]['pred'][0])
        t = [d['sent'] for d in final_dict[caption_name]['target']]
        for i in range(5):
#             target[i].append(t[i] if i < len(t) else '')
            target[i].append(t[i])
    
    eval_model = NLGEval(no_glove=True, no_skipthoughts=True)
    res = eval_model.compute_metrics(target, model_generate)
    print(res)
    return res



def eval_diversity_entity(final_dict):
    captions = {}
    for entity_id in final_dict:
        captions[entity_id] = final_dict[entity_id]['pred']

    LSA(captions)
    cider_LSA(captions)
    return None

def LSA(img_captions):
    """
    Codes from paper "Describing like humans: on diversity in image captioning. CVPR, 2019"
    Github link: https://github.com/qingzwang/DiversityMetrics/blob/master/pycocoevalcap/diversity_eval.py
    :param: img_captions is dict, format is {img_id: [captions str]}
    :return: each entities' score and the average diversity
    """
    def get_vocab(captions):
        """ Get the vocabulary of the captions """
        vocab = []
        for caption in captions:
            tokens = caption.split(' ')
            for token in tokens:
                if token not in vocab:
                    vocab.append(token)
        return vocab

    def term_document(captions):
        vocab = get_vocab(captions)
        term_doc = np.zeros([len(vocab), len(captions)])
        for doc_id in range(len(captions)):
            caption = captions[doc_id]
            tokens = caption.split(' ')
            for token in tokens:
                token_id = vocab.index(token)
                term_doc[token_id, doc_id] += 1
        return term_doc

    ratio = {}
    avg_diversity = 0
    for img_id in img_captions:
        captions = img_captions[img_id]
        print(captions)
        caption_num = len(captions)
        term_doc = term_document(captions)
        u, s, v = np.linalg.svd(term_doc)
        r = max(s) / s.sum()
        ratio[img_id] = -np.log10(r) / np.log10(caption_num)
        avg_diversity += -np.log10(r) / np.log10(caption_num)
    avg_diversity /= len(ratio)
    print('LSA average diversity:', avg_diversity)
    return ratio, avg_diversity

def cider_LSA(img_captions):
    """
    Cider kernalized LSA, also is named self-cider in paper "Describing like humans: on diversity in image captioning"
    :param img_captions:
    :return:
    """
    # Set cider score, it will take some time
    with open('../mscoco/coco-train2014-df.p', 'rb') as f:
        df_file = pickle.load(f, encoding='latin1')
    cider = Cider(df_file=df_file)

    ratio = {}
    avg_diversity = 0
    for img_id in img_captions:
        captions = img_captions[img_id]
        caption_num = len(captions)
        cov = np.zeros([caption_num, caption_num])
        for i in range(len(captions)):
            for j in range(len(captions)):
                new_gts = {img_id: [captions[i]]}
                new_res = {img_id: [captions[j]]}
                score, scores = cider.compute_score(new_gts, new_res)
                cov[i, j] = score
                cov[j, i] = score
        u, s, v = np.linalg.svd(cov)
        s_sqrt = np.sqrt(s)
        r = max(s_sqrt) / s_sqrt.sum()
        ratio[img_id] = -np.log10(r) / np.log10(caption_num)
        avg_diversity += -np.log10(r) / np.log10(caption_num)
    avg_diversity /= len(ratio)
    print('Cider-LSA (self-cider) average diversity: ', avg_diversity)
    return ratio, avg_diversity
    
if __name__ == '__main__':
#     print(eval_one_sent('a b c', 'a b c'))
    eval_txt('generated_sent.txt', 'target_generated_sent.txt')
#     print(eval_sent_list(['he is cool', 'he is cool'], ['she is cool', 'he is cool']));

