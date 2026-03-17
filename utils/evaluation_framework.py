"""
Procedures for running a privacy evaluation on a generative model
"""

from numpy import where, mean

from utils.constants import *

def get_accuracy(guesses, labels, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pIn = sum([g == l for g,l in zip(guesses[idxIn], labels[idxIn])])/len(idxIn)
    pOut = sum([g == l for g,l in zip(guesses[idxOut], labels[idxOut])])/len(idxOut)
    return pIn, pOut


def get_tp_fp_rates(guesses, labels):
    labels_arr = labels.values
    guesses_arr = guesses.values
    
    targetIn  = where(labels_arr == LABEL_IN)[0]
    targetOut = where(labels_arr == LABEL_OUT)[0]
    
    return (sum(guesses_arr[targetIn]  == LABEL_IN) / len(targetIn),
            sum(guesses_arr[targetOut] == LABEL_IN) / len(targetOut))

def get_probs_correct(pdf, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pdf[pdf > 1.] = 1.
    return mean(pdf[idxIn]), mean(pdf[idxOut])


def get_mia_advantage(tp_rate, fp_rate):
    return tp_rate - fp_rate


def get_ai_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_util_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_prob_removed(before, after):
    idxIn = where(before == LABEL_IN)[0]
    return 1.0 - sum(after[idxIn]/len(idxIn))




