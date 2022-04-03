import copy

import nltk
from torch import nn


def tokenize(sentence, word2vec=None):
    if type(sentence) is not str:
        return []
    punctuations = ['.', '?', ',', '', '(', ')']
    raw_text = sentence.lower()
    words = nltk.word_tokenize(raw_text)
    words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words if word not in punctuations]
    if word2vec is None:
        return words
    return [word for word in words if word in word2vec]


def clones(module, number):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number)])


def is_noun(word):
    word_tuple = nltk.pos_tag(word)
    if word_tuple[1] in {'NN', 'NNS', 'NNP', 'NNPS'}:
        return True
    return False


def is_predicate(word):
    word_tuple = nltk.pos_tag(word)
    if word_tuple[1] in {'VB', 'VBD'}:
        return True
    return False


def get_stem(word):
    return nltk.PorterStemmer().stem_word(word)