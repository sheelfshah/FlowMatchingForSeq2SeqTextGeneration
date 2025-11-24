import os, sys, glob, json
import numpy as np
import argparse

# import torch
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

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
        rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
        dist1.append(distinct_n_gram([recover], 1))
            
    # P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
    # print('*'*30)
    # print('avg BLEU score', np.mean(bleu))
    # print('avg ROUGE-L score', np.mean(rougel))
    # # print('avg berscore', torch.mean(F1))
    # print('avg dist1 score', np.mean(dist1))
    # print('avg len', np.mean(avg_len))

    return np.mean(bleu), np.mean(rougel), np.mean(dist1), np.mean(avg_len)

