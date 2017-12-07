# coding: utf-8
"""CS585: Assignment 4

See README.md
"""

###  You may add to these imports for gensim and nltk.
from collections import Counter
from itertools import product
import numpy as np
# import scipy
import pandas as pd
import tflearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.rnn import MultiRNNCell, GRUCell
#####################################


EMBEDDING_SIZE = 311 #(300 for word2vec embeddings and 11 for extra features (POS,CHUNK,CAP))
MAX_DOCUMENT_LENGTH=30
MAX_WORD_LENGTH=15
num_classes=5

def download_data():
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')
    nltk.download('brown')

def read_data(filename):
    
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
    
    classList = np.unique(true_labels)

    cm = np.zeros((len(classList), len(classList)), dtype=int)
    cmdf = pd.DataFrame(cm, index = classList, columns = classList)

    for i in range(0, len(true_labels)):
        cmdf[pred_labels[i]][true_labels[i]] += 1

    return cmdf

def evaluate(confusion_matrix):
    
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
    
    return np.average([evaluation_matrix[x]["f1"] for x in evaluation_matrix.columns if not x == "O"])

def evaluate_combinations(train_data, test_data, w2v_model):
    
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

        
        test_dicts, test_labels = make_feature_dicts(test_data, w2v_model,
                                                     token=True,
                                                     caps=combi[0],
                                                     pos=combi[1],
                                                     chunk=combi[2],
                                                     context=combi[3],
                                                     w2v=combi[4])
        X_test = vec.transform(test_dicts)

        model = getModel()

        model.fit(X, labels,n_epoch=1,validation_set=(X_test,test_labels), show_metric=False, batch_size=200)
        preds=np.asarray(model.predict(X_test))

        
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


def length(target):
    used = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def cost(prediction, target ):
    target = tf.reshape(target, [-1, MAX_DOCUMENT_LENGTH, num_classes])
    prediction = tf.reshape(prediction, [-1, MAX_DOCUMENT_LENGTH, num_classes])
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(length(target), tf.float32)
    return tf.reduce_mean(cross_entropy)

def getModel():

    net = tflearn.input_data([None, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE])
    net = rnn.static_bidirectional_rnn(MultiRNNCell([GRUCell(20)]*3), MultiRNNCell([GRUCell(20)]*3), tf.unstack(tf.transpose(net, perm=[1, 0, 2])), dtype=tf.float32)  #256=num_hidden, 3=num_layers
    net = tflearn.dropout(net[0],0.5)
    net = tf.transpose(tf.stack(net), perm=[1, 0, 2])

    net = tflearn.fully_connected(net, MAX_DOCUMENT_LENGTH*num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',loss=cost)

    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

    return model


if __name__ == '__main__':

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
    f = open("output2.txt",'w')
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    print('training data shape: %s\n' % str(X.shape))
    f.write('training data shape: %s\n' % str(X.shape))
    #
    # clf = LogisticRegression()
    # clf.fit(X, labels)

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

    model = getModel()

    model.fit(X, labels,n_epoch=1,validation_set=(X_test,test_labels), show_metric=False, batch_size=200)
    preds=np.asarray(model.predict(X_test))

    # preds = clf.predict(X_test)

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
