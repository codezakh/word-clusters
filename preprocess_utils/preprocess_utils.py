from __future__ import print_function, absolute_import, division
import json
import itertools
import string
import logging
import copy
import collections
import sys
import random

from nltk.corpus import stopwords as _stopwords
from sklearn import (manifold, cluster, metrics, cross_validation,
        tree, grid_search, ensemble, naive_bayes, preprocessing)
from IPython.core.display import clear_output

import gensim as gs
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd


def is_acronym(word):
    """Determine whether a word is an acronym.
    
    Parameters
    -------------
    word:string
    
    Returns
    -------------
    :bool
        True if word is an acronym false if not.
    """
    if word.islower() or len(word)==1 or word.isdigit():
        return False
    elif any([char.isupper() for char in word[1:]]):
        return True
    elif word[0].isupper():
        return False
    else:
        raise ValueError('unmatched pattern in {}'.format(word))


def standardize_words(document, keys=None):
    """Lowercase any words that are not acronyms.
    
    Parameters
    -----------
    document : dict
    keys : iterable of strings
        Each key value must be an iterable of strings.
        
    Returns
    -----------
    :dict
        A copy of the original dictionary with `keys` modified.
    """
    doc_copy = copy.copy(document)
    for key in keys:
        doc_copy[key] = [word for word in doc_copy.get(key) if word]
        doc_copy[key] = [word.lower() if not is_acronym(word) else word for
                word in doc_copy.get(key)]
    return doc_copy

def drop_stopwords(document, keys=None, stopset=None):
    """Removes any occurences for words in `stopset` from `document`.
    
    Parameters
    -----------
    document : dict
    keys : iterable of strings
        Each key value must be an iterable of strings.
    stopset : set
        Contains stopwords.
        
    Returns
    -----------
    :dict
        A copy of the original dictionary with `keys` modified.
    """
    doc_copy = copy.copy(document)
    for key in keys:
        doc_copy[key] = [word for word in doc_copy.get(key) if word not in stopset]
    return doc_copy

def depunc(phrase, pset=set(string.punctuation)):
    return ''.join([char for char in phrase if char not in pset or char])

def apply_transform(in_obj, t_manifest):
    o_obj = copy.copy(in_obj)
    for key, value in t_manifest.iteritems():
        o_obj[key] = value(o_obj.get(key))
    return o_obj


def unroll_document(document):
        flat_doc = itertools.chain.from_iterable(
                [itertools.repeat(word_id,repeats) for word_id, repeats in document])
        return flat_doc


def reroll_document_w_map(flat_document, remap=None):
    rerolled_document = []
    for w_id, grp in itertools.groupby([remap[old_w_id] for old_w_id in flat_document]):
        rerolled_document.append((w_id, len(list(grp))))
    return rerolled_document


def custom_auc_scorer(estimator, X, y):
    pred = estimator.predict_log_proba(X)
    return metrics.roc_auc_score(y, pred[:,1]-pred[:,0])


def one_vs_rest_clf(X,y):
    class_performance_dict = dict()
    for label in np.unique(y):
        if np.count_nonzero(y==label) < 4:
            continue
        clear_output()
        print('label {}'.format(label))
        clf_search = grid_search.RandomizedSearchCV(ensemble.RandomForestClassifier(),
                {'min_weight_fraction_leaf':list(np.arange(0.1,0.5,0.1)),
                    'max_depth':scipy.stats.distributions.randint(3,50),
                    'min_samples_leaf':scipy.stats.distributions.randint(1,10),
                    'min_samples_split':scipy.stats.distributions.randint(1,10),
                    'n_estimators':scipy.stats.distributions.randint(5,50)},
                cv=None,
                n_jobs=3,
                random_state=np.random.RandomState(seed=0),
                n_iter=100,
                scoring=custom_auc_scorer)
        y_binary = np.zeros_like(y)
        y_binary[y==label] = 1
        clf_search.fit(X,y_binary)
        class_performance_dict[label] = clf_search
    return class_performance_dict


def best_estimator_word_display(estimator,index2word,topn=10,
        index2word2=None):
    imps = np.argsort(estimator.feature_importances_)[::-1]
    return [index2word[idx] for idx in imps[:topn]]


def approp_doc(doc):
    for tok in doc:
        if not tok.is_alpha:
            continue
        elif is_acronym(unicode(tok)) and not tok.is_punct and not tok.is_stop and not tok.like_num:
            yield unicode(tok)
        elif not tok.like_num and not tok.is_stop and not tok.is_punct and tok.is_alpha:
            yield tok.lemma_


def walk_dependencies(doc):
    walks = []
    for tok in doc:
        is_term = list(tok.children)
        if not is_term:
            tok_ = tok
            walk = []
            while tok_ is not tok_.head:
                walk.append(tok_)
                tok_ = tok_.head
            else:
                walk.append(tok_)
            walks.append(walk)
    return walks
