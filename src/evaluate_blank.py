from sentence_transformers import CrossEncoder, SentenceTransformer, InputExample
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from torch.utils.data import DataLoader
from LabelAccuracyEvaluator import LabelAccuracyEvaluator
import pickle
import seaborn as sns

def plot_confusion_matrix(cm, blanklabels, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    classes[0] = 'homonymy'
    blanklabels[2] = 'Cohyp. transfer'
    plt.yticks(np.arange(len(classes[:-1])), classes[:-1], rotation=45)
    plt.xticks(np.arange(len(blanklabels[:4])), blanklabels[:4], rotation=45)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
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


blank2class = {'Specialization':1, 'Generalization':2, 'Cohyponymic transfer':3, 'Auto-Antonym':4, 'Metaphor':5}


rel2class = ['negative','hyponym','hypernym','co-hyponym','antonym','metaphor']
model = CrossEncoder('ChangeIsKey/change-type-classifier',device='cuda')


examples = []
words = []
labels = []
hierarchy_examples = []
metaphor_examples = []
antonym_examples = []
void_line = {}

with open('blank_dataset.tsv') as f:
    for j, line in enumerate(f):
        if not j == 0:
            line = line.replace('\n','').split('\t')
            def1 = line[9]
            def2 = line[10]
            if len(def1) > 0 and len(def2) > 0:
                words.append(line[0])
                #examples.append((def2,def1))
                if line[6] in blank2class:
                    examples.append([def2, def1])
                    labels.append(blank2class[line[6]])
                else:
                    blank2class[line[6]] = len(blank2class) + 1
                    examples.append([def2, def1])
                    labels.append(blank2class[line[6]])
                    #labels.append(0)
                    #sbert_examples.append(InputExample(texts=[def2, def1], label=labels[-1]))
                void_line[j-1] = 0
            else:
                void_line[j-1] = 1

print(blank2class)
blanklabels = [k for k in blank2class]

predictions = np.argmax(model.predict(examples),axis=-1)
print(len(predictions))

with open('cross_predictions_blank.txt','w+') as f:
    next_p = 0
    for j in sorted(void_line.keys()):
        if not void_line[j]:
            f.write(f'{labels[next_p]}\t{rel2class[predictions[next_p]]}\n')
            next_p = next_p + 1
        else:
            f.write(f'\n')

cm = confusion_matrix(labels, predictions)[1:5,:5]
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

rel2class[0] = 'homonymy'
blanklabels[2] = 'Cohyp. transfer'

columns_except_first = cm[:, 1:]
first_column = cm[:, 0:1]
new_matrix = np.concatenate((columns_except_first, first_column), axis=1)

rel2class = rel2class[1:] + ['homonymy']

ax = sns.heatmap(new_matrix, annot=True, xticklabels=rel2class, yticklabels=blanklabels[:4], cbar=False, cmap=sns.light_palette("seagreen", as_cmap=True), linewidth=0.5)
ax.set(xlabel="Predicted", ylabel="LSC-CTD Benchmark")

plt.tight_layout()
plt.show()
plt.savefig('cfn.svg')