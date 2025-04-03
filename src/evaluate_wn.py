from sentence_transformers import CrossEncoder, SentenceTransformer, InputExample
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from torch.utils.data import DataLoader
from LabelAccuracyEvaluator import LabelAccuracyEvaluator
import pickle
import json
import seaborn as sns

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    classes[0] = 'homonymy'
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


Dclass = {'negative':0,'hyponym':1,'hypernym':2,'co-hyponym':3,'antonym':4}


rel2class = ['negative','hyponym','hypernym','co-hyponym','antonym']
model = CrossEncoder('ChangeIsKey/change-type-classifier',device='cuda')


examples = []
words = []
labels = []
hierarchy_examples = []
metaphor_examples = []
antonym_examples = []

with open('datasets/test.jsonl','r') as f:
    for line in f:
        line = json.loads(line)
        if not line['label'] == 'metaphor':
            examples.append([line['definition1'],line['definition2']])
            labels.append(Dclass[line['label']])




predictions = np.argmax(model.predict(examples),axis=-1)

with open('wn_predictions.txt','w+') as f:
    for j,p in enumerate(predictions):
        f.write(f'{labels[j]}\t{rel2class[p]}\n')

cm = confusion_matrix(labels, predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot non-normalized confusion matrix
#plt.figure()

rel2class[0] = 'homonym'
#plot_confusion_matrix(cnf_matrix, classes=rel2class, normalize=True)
ax = sns.heatmap(cm, annot=True, xticklabels=rel2class, yticklabels=rel2class, cmap=sns.light_palette("seagreen", as_cmap=True), linewidth=0.5)
ax.set(xlabel="Predicted", ylabel="Actual")

plt.tight_layout()
plt.show()
plt.savefig('cfn_wn.svg')