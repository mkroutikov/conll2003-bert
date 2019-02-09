import pickle
import logging
import collections
import math
from types import SimpleNamespace
import numpy as np
from .preprocess_bert import load_vocabs
from .bioes import entities, entities_jie_bioes
import torch.nn.functional as F
from .models import BidiLstmBertModel
from .viterbi import decode_bioes_logits


logging.basicConfig(level=logging.INFO)

datadir = 'data/conll2003-preprocessed-bert'
BERT_MODEL = 'bert-base-multilingual-cased'

vocabs = load_vocabs(datadir, BERT_MODEL)

for name, v in vocabs.items():
    print(name)

MAX_LEN = 60


import torch
import torch.utils.data

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

with open(f'{datadir}/eng.testa.pickle', 'rb') as f:
    data_testa = pickle.load(f)

with open(f'{datadir}/eng.testb.pickle', 'rb') as f:
    data_testb = pickle.load(f)


def collate(batch):
    maxlen = max(len(x['words']) for x in batch)

    batch_size = len(batch)

    words  = torch.zeros(batch_size, maxlen, dtype=torch.int64)  # zero is out padding index!
    labels = torch.zeros(batch_size, maxlen, dtype=torch.int64)
    for i,item in enumerate(batch):
        x = torch.tensor(item['words'], dtype=torch.int64)
        y = torch.tensor(item['labels'], dtype=torch.int64)
        assert x.size(0) == y.size(0)
        words[i,:x.size(0)] = x
        labels[i,:y.size(0)] = y

    if maxlen > MAX_LEN:
        # limit size to avoid OOM when training (less than 100 records in the set are longer than 70)
        words  = words[:,:MAX_LEN]
        labels = labels[:,:MAX_LEN]

    return words.to(DEVICE), labels.to(DEVICE)


def get_entities(labels, *, labels_vocab):
    labels = [labels_vocab.decode(i) for i in labels]
    labels = [x for x in labels if x != '<pad>']

    return set(entities_jie_bioes(labels))


LABELS = sorted(set(x[2:] for x in vocabs['labels'].values if x not in ('<unk>', '<pad>', 'O')))


def error_analysis(model, dataset, *, vocabs, verbose):
    labels_vocab=vocabs['labels']
    words_vocab=vocabs['subwords']

    def logits_factory_factory(logits, i):
        def logits_factory(t, label):
            # t+1 because we skip leading [CLS]
            return logits[i,t+1, labels_vocab.encode(label)]
        return logits_factory

    miss = 0
    hits = 0
    for words, labels in torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate):

        batch_size = words.size(0)

        mask = (words > 0).long()
        seqlens = mask.sum(dim=1)

        print(words.shape)
        x = model(words)

        pred = torch.argmax(x, dim=2)

        for i in range(batch_size):
            true_labels = labels[i, 1:seqlens[i]-1].tolist()
            _, pred_labels = decode_bioes_logits(seqlens[i].item()-2, logits_factory=logits_factory_factory(x, i), labels=LABELS)

            pred_entities = get_entities(pred_labels, labels_vocab=labels_vocab)
            true_entities = entities(pred_labels)
            if pred_labels != true_labels:
                if verbose:
                    wrds = [words_vocab.decode(i) for i in words[i, :seqlens[i]].tolist()]
                    true_lbls = [labels_vocab.decode(i) for i in true_labels]
                    pred_lbls = [labels_vocab.decode(i) for i in pred_labels]
                    print()
                    print(wrds)
                    print(true_lbls)
                    print(pred_lbls)
                miss += 1
            else:
                hits += 1

    accuracy = hits / (hits + miss + 1.e-6)
    print(f'Accuracy: {accuracy*100:8.4f}%')


model = torch.load('best-conll.model', map_location=DEVICE)
model.train(False)
error_analysis(model, data_testa, vocabs=vocabs, verbose=False)
error_analysis(model, data_testb, vocabs=vocabs, verbose=False)

