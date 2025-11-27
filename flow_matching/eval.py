import os, sys, glob, json
import numpy as np
import argparse

# import torch
from torchmetrics.text.rouge import ROUGEScore

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

from utils import main_dir, time_str
from flow_matching import FlowMatching


def get_bleu(recover, reference):
    return sentence_bleu([reference.split()], recover.split(), smoothing_function=SmoothingFunction().method4,)

def distinct_n_gram(hypn,n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams/total_ngrams)
    return  np.mean(dist_list)

    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams/total_ngrams
    return  dist_n

def eval(generation_dict, sos='[CLS]', eos='[SEP]', sep='[SEP]', pad='[PAD]', direct_input=False, direct_output=False):
    """
    Args:
        generation_dict: {"source": List[str], "reference": List[str], "recover": List[str]}
    Returns:
        bleu: float
        rougel: float
        dist1: float
        avg_len: float
    """
    def clean_text(text):
        return text.replace(eos, '').replace(sos, '').replace(sep, '').replace(pad, '').strip()

    sources = list(map(clean_text, generation_dict['source']))
    references = list(map(clean_text, generation_dict['reference']))
    recovers = list(map(clean_text, generation_dict['recover']))

    bleu = []
    rougel = []
    dist1 = []
    avg_len = []

    for source, reference, recover in zip(sources, references, recovers):
        if direct_input:
            recover = source
        elif direct_output:
            recover = reference

        avg_len.append(len(recover.split(' ')))
        bleu.append(get_bleu(recover, reference))
        rougel.append(ROUGEScore()(recover, reference)['rougeL_fmeasure'].tolist())
        dist1.append(distinct_n_gram([recover], 1))
            
    # P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
    # print('*'*30)
    # print('avg BLEU score', np.mean(bleu))
    # print('avg ROUGE-L score', np.mean(rougel))
    # # print('avg berscore', torch.mean(F1))
    # print('avg dist1 score', np.mean(dist1))
    # print('avg len', np.mean(avg_len))

    return np.mean(bleu), np.mean(rougel), np.mean(dist1), np.mean(avg_len)

if __name__ == '__main__':
    FM = FlowMatching(main_dir, time_str)
    results = {
        1: {},
        2: {},
        4: {},
        8: {},
        16: {},
        32: {},
        64: {},
        100: {},
        200: {},
        500: {},
        1000: {},
        2000: {}
    }
    for n in results.keys():
        val_gens = FM.create_generations("valid", FM.variables['ema_params'][0.9999], num_steps = n)
        bleu, rougel, dist1, avg_len = eval(val_gens)
        results[n]["valid"] = {"gens": val_gens}
        results[n]["valid"]["bleu"] = bleu
        results[n]["valid"]["rougel"] = rougel
        results[n]["valid"]["dist1"] = dist1
        results[n]["valid"]["avg_len"] = avg_len
        print(f"Num steps: {n}, BLEU: {bleu:.6f}, ROUGE-L: {rougel:.6f}, Dist1: {dist1:.6f}, AvgLen: {avg_len:.6f}")
        test_gens = FM.create_generations("test", FM.variables['ema_params'][0.9999], num_steps = n)
        bleu, rougel, dist1, avg_len = eval(test_gens)
        results[n]["test"] = {"gens": test_gens}
        results[n]["test"]["bleu"] = bleu
        results[n]["test"]["rougel"] = rougel
        results[n]["test"]["dist1"] = dist1
        results[n]["test"]["avg_len"] = avg_len
        print(f"Num steps: {n}, BLEU: {bleu:.6f}, ROUGE-L: {rougel:.6f}, Dist1: {dist1:.6f}, AvgLen: {avg_len:.6f}")
    
    json.dump(results, open(os.path.join(FM.main_dir, f"plots/eval_{FM.args.checkpoint_dir}.json"), 'w'))
    