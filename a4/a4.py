# coding: utf-8
"""CS585: Assignment 4

See README.md
"""

###  You may add to these imports for gensim and nltk.
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
#####################################


def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')
    nltk.download('brown')

def read_data(filename):
    """
    Read the data file into a list of lists of tuples.

    Each sentence is a list of tuples.
    Each tuple contains four entries:
    - the token
    - the part of speech
    - the phrase chunking tag
    - the named entity tag

    For example, the first two entries in the
    returned result for 'train.txt' are:

    > train_data = read_data('train.txt')
    > train_data[:2]
    [[('EU', 'NNP', 'I-NP', 'I-ORG'),
      ('rejects', 'VBZ', 'I-VP', 'O'),
      ('German', 'JJ', 'I-NP', 'I-MISC'),
      ('call', 'NN', 'I-NP', 'O'),
      ('to', 'TO', 'I-VP', 'O'),
      ('boycott', 'VB', 'I-VP', 'O'),
      ('British', 'JJ', 'I-NP', 'I-MISC'),
      ('lamb', 'NN', 'I-NP', 'O'),
      ('.', '.', 'O', 'O')],
     [('Peter', 'NNP', 'I-NP', 'I-PER'), ('Blackburn', 'NNP', 'I-NP', 'I-PER')]]
    """
    sentences = []

    with open(filename, "r") as f:
        sentence = []
        for line in f:
            # if line.startswith("-DOCSTART-"):
            #     continue
            tokens = line.split()
            if len(tokens) == 0 and len(sentence) != 0:
                sentences.append(sentence)
                sentence = []
            elif len(tokens) == 0 and len(sentence) == 0:
                continue
            else:
                sentence.append( tuple(tokens) )
        sentences.append(sentence)

    return sentences

def get_word_2_vec():
    sentences = brown.sents()
    w2v_model = Word2Vec(sentences, size=50, window=5, min_count=5)
    return w2v_model

def make_feature_dicts(data, w2v_model,
                       token=True,
                       caps=True,
                       pos=True,
                       chunk=True,
                       context=True,
                       w2v=True):
    """
    Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
    and a value of 1.
    Also returns a numpy array of NER tags (strings), one per token.

    See a3_test.

    The parameter flags determine which features to compute.
    Params:
    data.......the data returned by read_data
    token......If True, create a feature with key 'tok=X', where X is the *lower case* string for this token.
    caps.......If True, create a feature 'is_caps' that is 1 if this token begins with a capital letter.
               If the token does not begin with a capital letter, do not add the feature.
    pos........If True, add a feature 'pos=X', where X is the part of speech tag for this token.
    chunk......If True, add a feature 'chunk=X', where X is the chunk tag for this token
    context....If True, add features that combine all the features for the previous and subsequent token.
               E.g., if the prior token has features 'is_caps' and 'tok=a', then the features for the
               current token will be augmented with 'prev_is_caps' and 'prev_tok=a'.
               Similarly, if the subsequent token has features 'is_caps', then the features for the
               current token will also include 'next_is_caps'.
    Returns:
    - A list of dicts, one per token, containing the features for that token.
    - A numpy array, one per token, containing the NER tag for that token.
    """
    feature = {}
    sentences = data
    featureList = []
    NERTags = []

    for sentence in sentences:
        for i in range(0, len(sentence)):
            word = sentence[i]
            if ("-DOCSTART-" in word):
                continue
            feature = getFeatures(word, w2v_model, token, caps, pos, chunk, w2v)

            if context:
                if i > 0 :
                    tempFeature = getFeatures(sentence[i-1], w2v_model, token, caps, pos, chunk, w2v)
                    if not "tok=-docstart-" in tempFeature:
                        for k,v in tempFeature.items():
                            feature["prev_"+k] = v

                if i < len(sentence) - 1:
                    tempFeature = getFeatures(sentence[i+1], w2v_model, token, caps, pos, chunk, w2v)
                    if not "tok=-docstart-" in tempFeature:
                        for k,v in tempFeature.items():
                            feature["next_"+k] = v

            featureList.append(feature)
            NERTags.append(word[3])

    return featureList, np.array(NERTags)


def getFeatures(word, w2v_model, token=True,
                       caps=True,
                       pos=True,
                       chunk=True,
                       w2v=True):
    feature = {}
    if token:
        feature["tok="+word[0].lower()] = 1
    if caps and word[0][0].isupper():
        feature["is_caps"] = 1
    if pos:
        feature["pos="+word[1]] = 1
    if chunk:
        feature["chunk="+word[2]] = 1
    if w2v:
        try:
            vec = w2v_model[word]
        except KeyError:
            vec = np.zeros((50, )).tolist()
        for i in range(0, len(vec)):
            feature["w2v_"+str(i)+"="+str(vec[i])] = 1
    return feature


def confusion(true_labels, pred_labels):
    """
    Create a confusion matrix, where cell (i,j)
    is the number of tokens with true label i and predicted label j.

    Params:
      true_labels....numpy array of true NER labels, one per token
      pred_labels....numpy array of predicted NER labels, one per token
    Returns:
    A Pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
    See Log.txt for an example.
    """
    classList = np.unique(true_labels)

    cm = np.zeros((len(classList), len(classList)), dtype=int)
    cmdf = pd.DataFrame(cm, index = classList, columns = classList)

    for i in range(0, len(true_labels)):
        cmdf[pred_labels[i]][true_labels[i]] += 1

    return cmdf

def evaluate(confusion_matrix):
    """
    Compute precision, recall, f1 for each NER label.
    The table should be sorted in ascending order of label name.
    If the denominator needed for any computation is 0,
    use 0 as the result.  (E.g., replace NaNs with 0s).

    NOTE: you should implement this on your own, not using
          any external libraries (other than Pandas for creating
          the output.)
    Params:
      confusion_matrix...output of confusion function above.
    Returns:
      A Pandas DataFrame. See Log.txt for an example.
    """
    NERTags = confusion_matrix.axes[1].values
    rowSum = confusion_matrix.sum(0)
    colSum = confusion_matrix.sum(1)

    data = {}
    data["precision"] = [confusion_matrix[t][t]/rowSum[t] for t in NERTags]
    data["recall"] = [confusion_matrix[t][t]/colSum[t] for t in NERTags]
    data["f1"] = [ (2*data["precision"][i]*data["recall"][i])/(data["precision"][i]+data["recall"][i])for i in range(0, len(NERTags))]

    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = NERTags

    return df

def average_f1s(evaluation_matrix):
    """
    Returns:
    The average F1 score for all NER tags,
    EXCLUDING the O tag.
    """
    return np.average([evaluation_matrix[x]["f1"] for x in evaluation_matrix.columns if not x == "O"])

def evaluate_combinations(train_data, test_data, w2v_model):
    """
    Run 16 different settings of the classifier,
    corresponding to the 16 different assignments to the
    parameters to make_feature_dicts:
    caps, pos, chunk, context
    That is, for one setting, we'll use
    token=True, caps=False, pos=False, chunk=False, context=False
    and for the next setting we'll use
    token=True, caps=False, pos=False, chunk=False, context=True

    For each setting, create the feature vectors for the training
    and testing set, fit a LogisticRegression classifier, and compute
    the average f1 (using the above functions).

    Returns:
    A Pandas DataFrame containing the F1 score for each setting,
    along with the total number of parameters in the resulting
    classifier. This should be sorted in descending order of F1.
    (See Log.txt).

    Note1: You may find itertools.product helpful for iterating over
    combinations.

    Note2: You may find it helpful to read the main method to see
    how to run the full analysis pipeline.
    """
    ###TODO
    inputs = product([False, True], repeat=5)
    columns = ["f1", "n_params", "caps", "pos", "chunk", "context", "w2v"]
    df = pd.DataFrame(columns=columns)
    df = df.fillna(0) # with 0s rather than NaNs

    for combi in inputs:

        dicts, labels = make_feature_dicts(train_data, w2v_model,
                                       token=True,
                                       caps=combi[0],
                                       pos=combi[1],
                                       chunk=combi[2],
                                       context=combi[3],
                                       w2v=combi[4])
        vec = DictVectorizer()
        X = vec.fit_transform(dicts)

        clf = LogisticRegression()
        clf.fit(X, labels)

        test_dicts, test_labels = make_feature_dicts(test_data, w2v_model,
                                                     token=True,
                                                     caps=combi[0],
                                                     pos=combi[1],
                                                     chunk=combi[2],
                                                     context=combi[3],
                                                     w2v=combi[4])
        X_test = vec.transform(test_dicts)

        preds = clf.predict(X_test)

        confusion_matrix = confusion(test_labels, preds)

        evaluation_matrix = evaluate(confusion_matrix)
        f1 = average_f1s(evaluation_matrix)
        print(5*len(clf.coef_[0]))

        data = {}
        data["f1"] = f1
        data["n_params"] = 5*len(clf.coef_[0])
        data["caps"] = combi[0]
        data["pos"] = combi[1]
        data["chunk"] = combi[2]
        data["context"] = combi[3]
        data["w2v"] = combi[4]

        df = df.append(data, ignore_index=True)
    df.sort_values("f1", axis=0, ascending=False, inplace=True)
    df.n_params = df.n_params.astype(np.int64)
    return df


if __name__ == '__main__':
    """
    This method is done for you.
    See Log.txt for expected output.
    """

    download_data()
    train_data = read_data('train.txt')
    w2v_model = get_word_2_vec()
    dicts, labels = make_feature_dicts(train_data, w2v_model,
                                   token=True,
                                   caps=True,
                                   pos=True,
                                   chunk=True,
                                   context=True,
                                   w2v=True)
    f = open("output1.txt",'w')
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    print('training data shape: %s\n' % str(X.shape))
    f.write('training data shape: %s\n' % str(X.shape))
    clf = LogisticRegression()
    clf.fit(X, labels)

    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data, w2v_model,
                                                 token=True,
                                                 caps=True,
                                                 pos=True,
                                                 chunk=True,
                                                 context=True,
                                                 w2v=True)
    X_test = vec.transform(test_dicts)
    print('testing data shape: %s\n' % str(X_test.shape))
    f.write('testing data shape: %s\n' % str(X_test.shape))

    preds = clf.predict(X_test)

    confusion_matrix = confusion(test_labels, preds)
    print('confusion matrix:\n%s\n' % str(confusion_matrix))
    f.write('confusion matrix:\n%s\n' % str(confusion_matrix))

    evaluation_matrix = evaluate(confusion_matrix)
    print('evaluation matrix:\n%s\n' % str(evaluation_matrix))
    f.write('evaluation matrix:\n%s\n' % str(evaluation_matrix))

    print('average f1s: %f\n' % average_f1s(evaluation_matrix))
    f.write('average f1s: %f\n' % average_f1s(evaluation_matrix))

    combo_results = evaluate_combinations(train_data, test_data, w2v_model)
    print('combination results:\n%s' % str(combo_results))
    f.write('combination results:\n%s' % str(combo_results))

    f.close()
