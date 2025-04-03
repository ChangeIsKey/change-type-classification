import argparse
import logging
import os
import random
import time
import json
import tempfile
import shutil
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (precision_recall_fscore_support, classification_report, accuracy_score)
from collections import namedtuple
from scipy.stats import spearmanr
from scipy.sparse import hstack
import torch
import random
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_data_multitask(path, return_sentences=False, return_inputs=False):
    examples = []
    sentences = [[],[]]
    rel2class = {'negative':0, 'hyponym':1, 'hypernym':2, 'co-hyponym':3, 'antonym':4}#, 'metaphor':5}
    labels = []
    with open(path) as f:
        for line in f:
            line = json.loads(line)
            if not line['label'] == 'metaphor':
                sentence1 = line['definition1']
                sentence2 = line['definition2']
                labels.append(rel2class[line['label']])
                sentences[0].append(sentence1)
                sentences[1].append(sentence2)
    return sentences, labels



sentences, y_train = load_data_multitask('datasets/train.jsonl')
feature_extraction = TfidfVectorizer(stop_words='english')
X = feature_extraction.fit_transform(sentences[0]+sentences[1])

X1 = X[:int(X.shape[0]/2),:]
X2 = X[int(X.shape[0]/2):,:]
X = hstack((X1,X2))

sentences, y_test = load_data_multitask('datasets/test.jsonl')
X_test = feature_extraction.transform(sentences[0]+sentences[1])


X1 = X_test[:int(X_test.shape[0]/2),:]
X2 = X_test[int(X_test.shape[0]/2):,:]
X_test = hstack((X1,X2))

#clf = SVC(probability=True, verbose=2, kernel='rbf')
#clf.fit(X, y_train)

clf = SGDClassifier(verbose=2,class_weight='balanced')
clf.fit(X, y_train)


result = clf.predict(X_test)

with open('svm_results.txt','w+') as f:
    f.write(classification_report(y_test,result))


cm = confusion_matrix(y_test,result)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot non-normalized confusion matrix
#plt.figure()

rel2class = ['negative','hyponym','hypernym','co-hyponym','antonym']

rel2class[0] = 'homonym'
ax = sns.heatmap(cm, annot=True, xticklabels=rel2class, yticklabels=rel2class, cbar=False, cmap=sns.light_palette("seagreen", as_cmap=True), linewidth=0.5)
ax.set(xlabel="Predicted", ylabel="Actual")

plt.tight_layout()
plt.show()
plt.savefig('svm_cfn_wn.svg')

plt.clf()


blank2class = {'Specialization':1, 'Generalization':2, 'Cohyponymic transfer':3, 'Auto-Antonym':4, 'Metaphor':5}

sentences = [[],[]]
words = []
labels = []
void_line = {}

with open('dataset_blank.tsv') as f:
    for j, line in enumerate(f):
        if not j == 0:
            line = line.replace('\n','').split('\t')
            def1 = line[9]
            def2 = line[10]
            if len(def1) > 0 and len(def2) > 0:
                words.append(line[0])
                sentences[0].append(def2)
                sentences[1].append(def1)
                if line[6] in blank2class:
                    labels.append(blank2class[line[6]])
                else:
                    blank2class[line[6]] = len(blank2class) + 1
                    labels.append(blank2class[line[6]])
                void_line[j-1] = 0
            else:
                void_line[j-1] = 1

X_test = feature_extraction.transform(sentences[0]+sentences[1])


X1 = X_test[:int(X_test.shape[0]/2),:]
X2 = X_test[int(X_test.shape[0]/2):,:]
X_test = hstack((X1,X2))

result = clf.predict(X_test)


blanklabels = [k for k in blank2class]

cm = confusion_matrix(labels, result)[1:5,:5]
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

blanklabels[2] = 'Cohyp. transfer'

columns_except_first = cm[:, 1:]
first_column = cm[:, 0:1]
new_matrix = np.concatenate((columns_except_first, first_column), axis=1)

print(rel2class)
rel2class = rel2class[1:] + ['homonymy']
print(rel2class)

ax = sns.heatmap(new_matrix, annot=True, xticklabels=rel2class, yticklabels=blanklabels[:4], cbar=False, cmap=sns.light_palette("seagreen", as_cmap=True), linewidth=0.5)
ax.set(xlabel="Predicted", ylabel="LSC-CTD Benchmark")

plt.tight_layout()
plt.show()
plt.savefig('svm_cfn.svg')