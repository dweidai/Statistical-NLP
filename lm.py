#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus, isTrigram):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus, isTrigram))

    def entropy(self, corpus, isTrigram):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s, isTrigram)
        return -(1.0/num_words)*(sum_logprob)
    
    
    def logprob_sentence(self, sentence, isTrigram):
        p = 0.0
        if(isTrigram):
            ## for unigram
            for i in xrange(len(sentence)):
                p += self.cond_logprob(sentence[i], sentence[:i])
            p += self.cond_logprob('END_OF_SENTENCE', sentence)
        else:
            # for unigram
            for i in xrange(len(sentence)):
                p += self.cond_logprob(sentence[i], sentence[:i])
            p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Trigram(LangModel): #with laplace
    def __init__(self, backoff = 0.0008):
        self.model = dict() # it will take in tri and bi and uni gram
        self.lbackoff = log(backoff, 2)
        self.alpha = 0.2 #used for lapalce smoothing
        self.total = 0 #used for laplace as well |V|
    
    ''' the goal of the inc_word is to count the number of tuples or appeerances of all the cases'''
    def inc_word(self, w, previous1, previous2):
        tri = (w, previous1, previous2)
        bi = (previous1, previous2)
        tri_tuple = tuple(tri)
        bi_tuple = tuple(bi)
        uni = w
        #this is the trigram case
        if tri in self.model:
            self.model[tri] += 1.0
        else:
            self.model[tri] = 1.0
        #this is the bigram case
        if bi in self.model:
            self.model[bi] += 1.0
        else:
            self.model[bi] = 1.0
        #this is the unigram case
        if uni in self.model:
            self.model[uni] += 1.0
        else:
            self.model[uni] = 1.0

    ''' the goal of fit_sentence is store all the information of the sentence into tuples'''
    def fit_sentence(self, sentence):
        sentence = ['*', '*'] + sentence + ['END_OF_SENTENCE']
        length = len(sentence) # this stores the length of the current sentence
        if length > 1:
            #stores the last case
            self.inc_word('END_OF_SENTENCE', sentence[length-1], sentence[length-2], )
        for word in range(2, length):
            self.inc_word(sentence[word], sentence[word -1], sentence[word - 2], ) #here is the regular one

    ''' the goal besides to calculate norm is also to count the number of words V as well'''
    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot
        self.total = len(self.vocab())
    
    ''' here is doing the laplace smoothing'''
    def cond_logprob(self, word, prev):
        prev_temp = prev
        prev_temp.insert(0,'*')
        prev_temp.insert(0,'*')
        previous1, previous2 = prev[-2:]
        tri = (word, previous1, previous2)
        bi = (previous1, previous2)
        tri_tuple = tuple(tri)
        bi_tuple = tuple(bi)
        if tri_tuple in self.model:
            return (self.model[tri] + self.alpha) / (self.model[bi] + self.alpha * self.total)
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()


class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

