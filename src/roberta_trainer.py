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
from torch.utils.data import DataLoader, IterableDataset
from sentence_transformers.models import Dropout
from sentence_transformers import SentenceTransformer, models, losses, InputExample, CrossEncoder
from LabelAccuracyEvaluator import LabelAccuracyEvaluator
from sentence_transformers.evaluation import BinaryClassificationEvaluator, EmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CESoftmaxAccuracyEvaluator
import torch
import random
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):

    def __init__(self, labels, n_classes, n_samples):
        self.labels_list = []
        for label in labels:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.labels_list):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.labels_list) // self.batch_size


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class Baseline:

    def __init__(self,args):
        self.args = args
        self.lr = float(args.lr)
        self.n_epochs = int(args.n_epochs)
        self.batch_size = int(args.batch_size)
        self.device = args.device
        self.weight_decay = float(args.weight_decay)
        self.output_path = args.output_path
        self.do_validation = args.do_validation
        self.train_path = args.train_path
        self.dev_path = args.dev_path
        self.pretrained_model = args.pretrained_model
        self.max_seq_length = int(args.max_seq_length)
        self.warmup_percentage = float(args.warmup_percentage)
        self.loss = args.loss
        self.evaluation = args.evaluation
        self.finetune_sbert = args.finetune_sbert
        self.sbert_pretrained_model = args.sbert_pretrained_model
        self.pretrained_model = args.pretrained_model
        self.model_type = args.model_type
        self.remove_sentence = args.remove_sentence
        self.dropout = args.dropout


    def load_data_multitask(self, path, return_sentences=False, return_inputs=False):
        examples = []
        sentences = [[],[]]
        rel2class = {'negative':0, 'hyponym':1, 'hypernym':2, 'co-hyponym':3, 'antonym':4, 'positive':5}#, 'metaphor':5}
        labels = []
        with open(path) as f:
            for line in f:
                line = json.loads(line)
                if not 'label' in line:
                    sentence1 = line['definition1']
                    sentence2 = line['definition2']
                    examples.append(InputExample(texts=[sentence1, sentence2], label=rel2class['positive']))
                    labels.append(rel2class['positive'])
                    sentences[0].append(sentence1)
                    sentences[1].append(sentence2)
                elif not line['label'] == 'metaphor':
                    sentence1 = line['definition1']
                    sentence2 = line['definition2']
                    examples.append(InputExample(texts=[sentence1, sentence2], label=rel2class[line['label']]))
                    labels.append(rel2class[line['label']])
                    sentences[0].append(sentence1)
                    sentences[1].append(sentence2)
                
                """
                if line['label'] == 'co-hyponym' or line['label'] == 'antonym' or line['label'] == 'metaphor':
                    examples.append(InputExample(texts=[sentence2, sentence1], label=rel2class[line['label']]))
                    labels.append(rel2class[line['label']])
                    sentences[0].append(sentence2)
                    sentences[1].append(sentence1)
                """

        batch_loader = BalancedBatchSampler(labels,len(set(labels)),int(self.batch_size/len(set(labels))))
        dataloader = DataLoader(examples)#, batch_sampler=batch_loader) 

        if return_sentences:
            return sentences, labels
        elif return_inputs:
            return None
        else:
            return dataloader



    def init_model(self):
        if self.finetune_sbert:
            if self.model_type == 'crossencoder':
                self.model = CrossEncoder(self.sbert_pretrained_model,num_labels=6)
            else:
                self.model = SentenceTransformer(self.sbert_pretrained_model)
                self.loss = losses.SoftmaxLoss(model=self.model, sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(), num_labels=6)
        else:
            if self.model_type == 'crossencoder':
                self.model = CrossEncoder(self.pretrained_model,num_labels=6)
            else:
                word_embedding_model = models.Transformer(self.pretrained_model, max_seq_length=self.max_seq_length)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                if self.dropout:
                    self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)
                else:
                    self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, Dropout(0.2)], device=self.device)

                self.loss = losses.SoftmaxLoss(model=self.model, sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(), num_labels=6)

    def train(self):

        class Callback(object):
            def __init__(self, warmup_steps, steps_per_epoch, model_name, output_path, baseline_model):
                self.best_score = 0
                self.patience = 10
                self.warmup_steps = warmup_steps
                self.steps_per_epoch = steps_per_epoch
                self.model_name = model_name
                self.output_path = output_path
                self.baseline_model = baseline_model

            def __call__(self, score, epoch, steps):
                if score > self.best_score:        
                    self.best_score = score
                else:
                    if max(0, (self.steps_per_epoch * (epoch - 1))) + steps > warmup_steps:
                        if self.patience == 0:
                            print('Early stop training.')
                            with open(os.path.join(f"{self.output_path}",f"results.tsv"), 'a+') as f:
                                f.write(f'{self.model_name}\t{self.best_score}\n')
                            exit()
                        self.patience = self.patience - 1
                    if self.baseline_model.model_type == 'siamese':
                        with open(os.path.join(os.path.join(self.output_path,self.model_name),"loss.pkl"), 'wb+') as f:
                            pickle.dump(self.baseline_model.loss, f)

        self.init_model()
        train_dataloader = self.load_data_multitask(self.train_path)
        train_objectives = [[train_dataloader,self.loss]]

        len_train_data = len(train_dataloader)
        warmup_steps = self.warmup_percentage * (len_train_data * self.n_epochs)
        evaluation_steps = int(0.25 * (len_train_data))


        if self.evaluation == 'multi-task':
            if self.model_type == 'siamese':
                dev_dataloader = self.load_data_multitask(self.dev_path)
                evaluator = LabelAccuracyEvaluator(dev_dataloader, 'multi-task', self.loss, 6)
            else:
                sentences, labels = self.load_data_multitask(self.dev_path, return_sentences=True)
                sentences = [[sentences[0][j], sentences[1][j]] for j in range(len(sentences[0]))]
                evaluator = CESoftmaxAccuracyEvaluator(sentences, labels)

        if self.model_type == 'siamese':
            self.model.fit(
                train_objectives=train_objectives,
                epochs=self.n_epochs,
                optimizer_params={'lr': self.lr},
                warmup_steps=warmup_steps,
                evaluator=evaluator,
                checkpoint_save_steps=None,
                callback=Callback(warmup_steps, len_train_data,model_name=f"model_{self.lr}_{self.weight_decay}", output_path=self.output_path, baseline_model=self),
                evaluation_steps=evaluation_steps,
                output_path=os.path.join(f"{self.output_path}",f"model_{self.lr}_{self.weight_decay}"),
                weight_decay=self.weight_decay,
                show_progress_bar=False
            )
        else:
            self.model.fit(
                train_dataloader=train_dataloader,
                epochs=self.n_epochs,
                optimizer_params={'lr': self.lr},
                warmup_steps=warmup_steps,
                evaluator=evaluator,
                callback=Callback(warmup_steps, len(train_dataloader),model_name=f"model_{self.lr}_{self.weight_decay}", output_path=self.output_path, baseline_model=self),
                evaluation_steps=evaluation_steps,
                output_path=os.path.join(f"{self.output_path}",f"model_{self.lr}_{self.weight_decay}"),
                weight_decay=self.weight_decay,
                show_progress_bar=True
            )

        with open(os.path.join(f"{self.output_path}",f"model_{self.lr}_{self.weight_decay}","cmd_args.txt"), 'w+') as f:
            json.dump(self.args.__dict__, f, indent=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Baseline model',
        description="Training of the baseline model")

    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--n_epochs', default=50)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--weight_decay', default=0.0)
    parser.add_argument('--warmup_percentage', default=0.1)
    parser.add_argument('--loss', default="contrastive", choices=['ce', 'softmax', 'mse', 'cosine', 'contrastive', 'triplet'])
    parser.add_argument('--remove_sentence', action="store_true")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train_path', default="")
    parser.add_argument('--evaluation', default="binary")
    parser.add_argument('--strategy', default='target', choices=['context', 'target'])
    parser.add_argument('--dev_path', default="")
    parser.add_argument('--test_path', default="")
    parser.add_argument('--model_type', default="siamese")
    parser.add_argument('--do_validation', default=True)
    parser.add_argument('--output_path', default='models')
    parser.add_argument('--finetune_sbert', action="store_true")
    parser.add_argument('--pretrained_model', default='roberta-large')
    parser.add_argument('--sbert_pretrained_model', default='paraphrase-multilingual-mpnet-base-v2')
    parser.add_argument('--max_seq_length', default=512)
    parser.add_argument('--dropout', action="store_true")


    args = parser.parse_args()

    set_seed()

    b = Baseline(args)
    b.train()
