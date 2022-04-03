from collections import Counter

import nltk
import torch
from torch import nn
import numpy as np


def tokenize(sentence, use_prototype=False, word2vec=None):
    if type(sentence) is not str:
        return []
    punctuations = ['.', '?', ',', '', '(', ')', '!', ':', '…']
    raw_text = sentence.lower()
    words = nltk.word_tokenize(raw_text)
    if use_prototype:
        words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words if word not in punctuations]
    else:
        words = [word for word in words if word not in punctuations]
    if word2vec is None:
        return words
    return [word for word in words if word in word2vec]


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


class Vocabulary(object):
    def __init__(self, sentences, vocab_size=10000):
        super(Vocabulary, self).__init__()
        self.word2ind = {
            '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<MASK>': 4
        }
        self.ind2word = {
            0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>', 4: '<MASK>'
        }
        word_list = []
        for sentence in sentences:
            word_list.extend(tokenize(sentence))
        vocab_counter = Counter(word_list)
        vocab_count = 5
        vocab_list = vocab_counter.most_common(vocab_size - vocab_count)
        for vocab in vocab_list:
            self.word2ind[vocab[0]] = vocab_count
            self.ind2word[vocab_count] = vocab[0]
            vocab_count += 1
        self.max_word_id = max(list(self.ind2word.keys()))
        self.word_num = self.max_word_id + 1

    def digitize(self, sentences):
        if type(sentences[0]) is str:
            sentence = sentences
            return [self.stoi(word) for word in tokenize(sentence)]
        return [[self.stoi(word) for word in tokenize(sentence)] for sentence in sentences]

    def pad(self, digitized_sentences, max_len):
        if type(digitized_sentences[0]) is int:
            sentence = digitized_sentences
            return sentence + [self.stoi("<PAD>")] * (max_len - len(sentence))
        return [(sentence + [self.stoi("<PAD>")] * (max_len - len(sentence))) for sentence in digitized_sentences]

    def score2sentence(self, score):
        # score in shape (max_len, vocab_size)
        _, words_idx = torch.max(score, dim=-1)
        sentence = []
        for word_idx in words_idx:
            sentence.append(self.itos(int(word_idx)))
        return ' '.join(sentence)

    def id2sentence(self, id):
        sentence = []
        for word_idx in id:
            sentence.append(self.itos(int(word_idx)))
        return ' '.join(sentence)

    def stoi(self, word):
        if word in self.word2ind:
            return self.word2ind[word]
        else:
            return self.word2ind['<UNK>']

    def itos(self, index):
        if index in self.ind2word:
            return self.ind2word[index]
        else:
            return '<UNK>'

    def itoa(self, index):
        if index > self.max_word_id:
            return self.itoa(self.stoi('<UNK>'))
        one_hot = np.zeros(self.max_word_id)
        one_hot[index] = 1
        return one_hot

    def stoa(self, word):
        return self.itoa(self.stoi(word))

    @property
    def MASK(self):
        return self.stoi('<MASK>')

    @property
    def PAD(self):
        return self.stoi('<PAD>')


if __name__ == "__main__":
    result = tokenize("Whether to repeat the iterator for multiple epochs. Default: False.")
    sentences = [[0, 1, 2, 3, 4, 5]]
    print([sentence + [0] * (25 - len(sentence)) for sentence in sentences])
    print(result)
