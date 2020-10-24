from nlgeval import compute_metrics
from nlgeval import compute_individual_metrics
from nlgeval import NLGEval

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
    Input is List[str], List[str]
    
    One example:
    {'Bleu_1': 0.8333333327777782, 'Bleu_2': 0.7886751340033071, 'Bleu_3': 0.5000034663341466, 'Bleu_4': 0.015813524723354844, 'METEOR': 0.6910678648089302, 'ROUGE_L': 0.8333333333333333, 'CIDEr': 0.0}
    """
    eval_model = NLGEval(no_glove=True, no_skipthoughts=True)
    res = {}
    for m_sent, t_sent in zip(model_sent_list, target_sent_list):
        temp_res = eval_model.compute_individual_metrics(hyp=m_sent, ref=[t_sent])
        for key in temp_res:
            res[key] = res.get(key, 0) + temp_res[key]
    for key in res:
        res[key] = res[key] / len(model_sent_list)
    return res
    
if __name__ == '__main__':
#     print(eval_one_sent('a b c', 'a b c'))
    eval_txt('generated_sent.txt', 'target_generated_sent.txt')
#     print(eval_sent_list(['he is cool', 'he is cool'], ['she is cool', 'he is cool']));
