# coding: utf-8
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request

class HMM:
    def __init__(self, smoothing=0):
        """
        Construct an HMM model with add-k smoothing.
        Params:
          smoothing...the add-k smoothing value

        This is DONE.
        """
        self.smoothing = smoothing
        # This is printing the value 0.001

    def fit_transition_probas(self, tags):
        """
        Estimate the HMM state transition probabilities from the provided data.

        Creates a new instance variable called `transition_probas` that is a
        dict from a string ('state') to a dict from string to float. E.g.
        {'N': {'N': .1, 'V': .7, 'D': .2},
         'V': {'N': .3, 'V': .5, 'D': .2},
         ...
        }
        See test_hmm_fit_transition.

        Params:
          tags...a list of lists of strings representing the tags for one sentence.
        Returns:
            None
        """
        tag_vocab = []
        present_pos = {}
        present_total_tx = {}

        for sen_tags in tags:
            
            for i in range(1, len(sen_tags)):
                
                if not sen_tags[i-1] in tag_vocab:
                    
                    tag_vocab.append(sen_tags[i-1])
                if not sen_tags[i] in tag_vocab:
                    tag_vocab.append(sen_tags[i])

                tx_counts = {}
                if not sen_tags[i-1] in present_pos:
                    tx_counts[sen_tags[i]] = 1
                else:
                    tx_counts = present_pos[sen_tags[i-1]]
                    if not sen_tags[i] in tx_counts:
                        tx_counts[sen_tags[i]] = 1
                    else:
                        tx_counts[sen_tags[i]] += 1
                present_pos[sen_tags[i-1]] = tx_counts

        for tag in present_pos.keys():
            tx_counts = present_pos.get(tag)
            sumR = 0
            sumR = np.array([v for v in tx_counts.values()]).sum()
            present_total_tx[tag] = sumR

        for tag in tag_vocab:
            if not tag in present_pos.keys():
                present_pos[tag] = {}
                present_total_tx[tag] = 0
            for tag1 in tag_vocab:
                tx_counts = present_pos[tag]
                if tag1 in tx_counts.keys():
                    tx_counts[tag1] += self.smoothing
                else:
                    tx_counts[tag1] = self.smoothing
                tx_counts[tag1] /= (present_total_tx[tag] + (self.smoothing*len(tag_vocab)))

        tag_vocab.sort()
        self.states = tag_vocab
        self.transition_probas = present_pos
        return

    def fit_emission_probas(self, sentences, tags):
        """
        Estimate the HMM emission probabilities from the provided data.

        Creates a new instance variable called `emission_probas` that is a
        dict from a string ('state') to a dict from string to float. E.g.
        {'N': {'dog': .1, 'cat': .7, 'mouse': 2},
         'V': {'run': .3, 'go': .5, 'jump': 2},
         ...
        }

        Params:
          sentences...a list of lists of strings, representing the tokens in each sentence.
          tags........a list of lists of strings, representing the tags for one sentence.
        Returns:
            None

        See test_hmm_fit_emission.
        """
        pos_map = {}
        pos_word_total = {}
        no_of_sentences = len(sentences)
        word_vocab = []
        for i in range(0, no_of_sentences):
            sentence = sentences[i]
            sen_tags = tags[i]

            for j in range(0, len(sentence)):
                word_map = {}
                if not sentence[j] in word_vocab:
                    word_vocab.append(sentence[j])

                if not sen_tags[j] in pos_map.keys():
                    word_map[sentence[j]] = 1
                else:
                    word_map = pos_map[sen_tags[j]]
                    if not sentence[j] in word_map.keys():
                        word_map[sentence[j]] = 1
                    else:
                        word_map[sentence[j]] += 1
                pos_map[sen_tags[j]] = word_map

        for tag in pos_map.keys():
            word_map = pos_map[tag]
            sumR = 0
            sumR = np.array([v for v in word_map.values()]).sum()
            pos_word_total[tag] = sumR

        for tag in pos_map.keys():
            word_map = pos_map[tag]
            for w in word_vocab:
                if w in word_map.keys():
                    word_map[w] += self.smoothing
                else:
                    word_map[w] = self.smoothing
                word_map[w] /= (pos_word_total[tag] + (self.smoothing*len(word_vocab)))

        self.emission_probas = pos_map
        return

    def fit_start_probas(self, tags):
        """
        Estimate the HMM start probabilities form the provided data.

        Creates a new instance variable called `start_probas` that is a
        dict from string (state) to float indicating the probability of that
        state starting a sentence. E.g.:
        {
            'N': .4,
            'D': .5,
            'V': .1
        }

        Params:
          tags...a list of lists of strings representing the tags for one sentence.
        Returns:
            None

        See test_hmm_fit_start
        """
        tag_vocab = self.states
        start_probas = {}
        no_of_sentences = len(tags)
        for sent_tags in tags:
            if not sent_tags[0] in start_probas.keys():
                start_probas[sent_tags[0]] = 1
            else:
                start_probas[sent_tags[0]] += 1

        for tag in tag_vocab:
            if tag in start_probas.keys():
                start_probas[tag] += self.smoothing
            else:
                start_probas[tag] = self.smoothing
            start_probas[tag] /= (no_of_sentences + (self.smoothing*len(tag_vocab)))

        self.start_probas = start_probas
        return

    def fit(self, sentences, tags):
        """
        Fit the parameters of this HMM from the provided data.

        Params:
          sentences...a list of lists of strings, representing the tokens in each sentence.
          tags........a list of lists of strings, representing the tags for one sentence.
        Returns:
            None

        DONE. This just calls the three fit_ methods above.
        """
        self.fit_transition_probas(tags)
        self.fit_emission_probas(sentences, tags)
        self.fit_start_probas(tags)

    def viterbi(self, sentence):
        """
        Perform Viterbi search to identify the most probable set of hidden states for
        the provided input sentence.

        Params:
          sentence...a lists of strings, representing the tokens in a single sentence.

        Returns:
          path....a list of strings indicating the most probable path of POS tags for
                  this sentence.
          proba...a float indicating the probability of this path.
        """

        states = self.states
        start_probas = self.start_probas
        trans_probas = self.transition_probas
        emiss_probas = self.emission_probas

        k = len(states)

        t = len(sentence)

        t1 = np.empty((k, t))
        t2 = np.empty((k, t))

        for i in range(0, k):

            t1[i, 0] = start_probas[states[i]] * emiss_probas[states[i]][sentence[0]]
            t2[i, 0] = 0

        for i in range(1, t):
            for j in range(0, k):
                max_prob = -1
                index = -1
                for z in range(0, k):
                    if (t1[z, i-1]*trans_probas[states[z]][states[j]]) > max_prob:
                        max_prob = t1[z, i-1]*trans_probas[states[z]][states[j]]
                        index = z
                t1[j, i] = emiss_probas[states[j]][sentence[i]] * max_prob
                t2[j, i] = index
        max_t1 = -1
        zt = -1
        for i in range(0, k):
            if (t1[i, t-1]) > max_t1:
                max_t1 = t1[i, t-1]
                zt = i
        z = []
        z.insert(0, zt)
        x = [states[zt]]

        for i in range(t-1, 0, -1):
            z.insert(0, t2[int(z[0]), i])
            x.insert(0, states[int(z[0])])

        return x, max_t1


def read_labeled_data(filename):
    """
    Read in the training data, consisting of sentences and their POS tags.

    Each line has the format:
    <token> <tag>

    New sentences are indicated by a newline. E.g. two sentences may look like this:
    <token1> <tag1>
    <token2> <tag2>

    <token1> <tag1>
    <token2> <tag2>
    ...

    See data.txt for example data.

    Params:
      filename...a string storing the path to the labeled data file.
    Returns:
      sentences...a list of lists of strings, representing the tokens in each sentence.
      tags........a lists of lists of strings, representing the POS tags for each sentence.
    """
    sentences = []
    tags = []

    with open(filename, "r") as f:
        sentence = []
        tag = []
        for line in f:
            tokens = line.split(None, 1)

            if len(tokens) == 0:
                sentences.append(sentence)
                tags.append(tag)
                sentence = []
                tag = []
            else:
                sentence.append(tokens[0])
                tag.append(tokens[1].replace("\n",""))

    return sentences, tags


def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')


if __name__ == '__main__':
    """
    Read the labeled data, fit an HMM, and predict the POS tags for the sentence
    'Look at what happened'

    DONE - please do not modify this method.

    The expected output is below. (Note that the probability may differ slightly due
    to different computing environments.)

    $ python3 a2.py
    model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
    predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
    (['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
    """
    fname = 'data.txt'
    if not os.path.isfile(fname):
        download_data()
    sentences, tags = read_labeled_data(fname)

    model = HMM(.001)
    model.fit(sentences, tags)
    print('model has %d states' % len(model.states))
    print(model.states)
    sentence = ['Look', 'at', 'what', 'happened']
    print('predicted parts of speech for the sentence %s' % str(sentence))
    print(model.viterbi(sentence))
