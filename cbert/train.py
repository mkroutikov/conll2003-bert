import pickle
import logging
import collections
import math
from types import SimpleNamespace
import numpy as np
from .preprocess import load_vocabs
from .glove import Glove
from .bioes import entities, entities_jie_bioes
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from .models import BidiLstmGloveModel


logging.basicConfig(level=logging.INFO)

datadir = 'data/conll2003-preprocessed'

glove = Glove.load('glove/glove.6B.100d.txt')
logging.info('Loaded %s Glove vectors', len(glove))

vocabs = load_vocabs(datadir)

for name, v in vocabs.items():
    print(name, len(v))

# build mapping from word index to glove vector, for all words in our vocab
words_vocab = vocabs['words']


def get_embedding(glove, word):
    if word in glove:
        return glove[word]
    elif word.lower() in glove:
        return glove[word.lower()]
    else:
        return get_random_embedding(glove.dim)

def get_random_embedding(dim):
    scale = math.sqrt(3./dim)
    return np.random.uniform(-scale, scale, size=(dim,))

glove_embedding = np.array([get_embedding(glove, x) for x in words_vocab.values])
#glove_embedding = np.array([get_random_embedding(glove.dim) for x in words_vocab.values])
#glove_embedding = np.array([glove[x] for x in words_vocab.values])
print(glove_embedding.shape)

import torch
import torch.utils.data

with open(f'{datadir}/eng.train.pickle', 'rb') as f:
    data_train = pickle.load(f)

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

    return words, labels


def get_entities(labels, *, labels_vocab):
    labels = [labels_vocab.decode(i) for i in labels]
    labels = [x for x in labels if x != '<pad>']

    return set(entities_jie_bioes(labels))

def update_stats(stats, predicted_labels, true_labels, *, labels_vocab):
    predicted_entities = get_entities(predicted_labels, labels_vocab=labels_vocab)
    true_entities = get_entities(true_labels, labels_vocab=labels_vocab)

    for e in predicted_entities & true_entities:
        stats[e.label, 'tp'] += 1
        stats['tp'] += 1

    for e in predicted_entities - true_entities:
        stats[e.label, 'fp'] += 1
        stats['fp'] += 1

    for e in true_entities - predicted_entities:
        stats[e.label, 'fn'] += 1
        stats['fn'] += 1

def display_stats(stats, labels):
    tp = 0
    fp = 0
    fn = 0
    for l in labels:
        prec = stats[l, 'tp'] / (stats[l, 'fp'] + stats[l, 'tp'] + 1.e-6)
        recall = stats[l, 'tp'] / (stats[l, 'fn'] + stats[l, 'tp'] + 1.e-6)
        f1 = 2 * prec * recall / (prec + recall + 1.e-6)
        print(f'{l:6}:: prec: {prec:6.4f}, recall: {recall:6.4f}, F1: {f1:6.4f}')
        tp += stats[l, 'tp']
        fp += stats[l, 'fp']
        fn += stats[l, 'fn']

    prec = stats['tp'] / (stats['fp'] + stats['tp'] + 1.e-6)
    recall = stats['tp'] / (stats['fn'] + stats['tp'] + 1.e-6)
    f1 = 2 * prec * recall / (prec + recall + 1.e-6)
    print(f'------:: prec: {prec:6.4f}, recall: {recall:6.4f}, F1: {f1:6.4f}')
    return f1


def evaluate(model, dataset, *, vocabs, lbls):
    stats = collections.defaultdict(int)
    losses = []
    total = 0
    hits = 0
    record_hits = 0
    total_records = 0
    for words, labels in torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate):

        batch_size = words.size(0)

        mask = (words > 0).long()
        seqlens = mask.sum(dim=1)
        longest_sequence = seqlens.max().item()

        x = model(words)

        pred = torch.argmax(x, dim=2)

        for i in range(batch_size):
            true_labels = labels[i, :seqlens[i]].tolist()
            pred_labels = pred[i, :seqlens[i]].tolist()

            update_stats(stats, pred_labels, true_labels, labels_vocab=vocabs['labels'])
            for pl,tl in zip(pred_labels, true_labels):
                total += 1
                if pl == tl:
                    hits += 1
            if all(x==y for x,y in zip(true_labels, pred_labels)):
                record_hits += 1
            total_records += 1

        x = x.transpose(1, 2)
        loss = F.cross_entropy(x, labels, reduction='mean')
        losses.append(loss.item())

    print(f'AVG loss: {sum(losses) / len(losses):6.4f}')
    print(f'AVG token accuracy: {hits/(total+1.e-6):6.4f}')
    print(f'AVG record accuracy: {record_hits/(total_records+1.e-6):6.4f}')
    return display_stats(stats, lbls)


def error_analysis(model, dataset, *, vocabs, verbose):
    labels_vocab=vocabs['labels']
    words_vocab=vocabs['words']

    miss = 0
    hits = 0
    for words, labels in torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate):

        batch_size = words.size(0)

        mask = (words > 0).long()
        seqlens = mask.sum(dim=1)
        longest_sequence = seqlens.max().item()

        words = words[:,:longest_sequence]
        mask = mask[:,:longest_sequence]
        labels = labels[:,:longest_sequence]

        x = model(words)

        pred = torch.argmax(x, dim=2)

        for i in range(batch_size):
            true_labels = labels[i, :seqlens[i]].tolist()
            pred_labels = pred[i, :seqlens[i]].tolist()

            pred_entities = get_entities(pred_labels, labels_vocab=labels_vocab)
            true_entities = get_entities(true_labels, labels_vocab=labels_vocab)
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


model = BidiLstmGloveModel(
    glove_embedding=glove_embedding,
    labels_vocab_size=len(vocabs['labels']),
    num_layers=1
)

optimizer = torch.optim.SGD(model.parameters(), lr=1.5, momentum=0.0, weight_decay=0.000001)

def decay_learning_rate(epoch):
    x = 1. / (1. + 0.05 * epoch)
    print(f'LR decay factor: {x:6.4f}')
    return x

LABELS = sorted(set(x[2:] for x in vocabs['labels'].values if x not in ('<unk>', '<pad>', 'O')))

schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, decay_learning_rate)

best = SimpleNamespace(f1=0.0, epoch=-1)
for epoch in range(100):
    schedule.step()

    stats = collections.defaultdict(int)
    count = 0
    for words, labels in torch.utils.data.DataLoader(data_train, batch_size=10, shuffle=True, collate_fn=collate):

        optimizer.zero_grad()

        batch_size = words.size(0)

        mask = (words > 0).long()
        seqlen = mask.sum(dim=1)

        x = model(words)
        pred = torch.argmax(x, dim=2)

        x = x.transpose(1, 2)
        loss = F.cross_entropy(x, labels, ignore_index=0, reduction='mean')

        loss.backward()
        optimizer.step()

        correct_toks = ((pred==labels).long() * mask).sum(dim=1)
        numtokens = seqlen.sum().item()
        stats['numtokens'] += numtokens
        stats['correct_tokens'] += correct_toks.sum().item()
        stats['numrecords'] += batch_size
        stats['correct_records'] += (correct_toks==seqlen).long().sum().item()
        stats['loss'] += loss.item()

        count += 1
        if count % 50 == 0:
            token_accuracy = stats['correct_tokens'] / (stats['numtokens'] + 1.e-6)
            record_accuracy = stats['correct_records'] / (stats['numrecords'] + 1.e-6)
            print(
                f'epoch: {epoch:5}, step: {count:6}, loss: {stats["loss"]/stats["numrecords"]:10.4f}, best_f1@epoch: {best.f1:6.4f}@{best.epoch}, '
                f'accuracy: {token_accuracy:6.4f}, record accuracy: {record_accuracy:6.4f}'
            )
    stats.clear()

    model.train(False)
    f1 = evaluate(model, data_testa, vocabs=vocabs, lbls=LABELS)
    if f1 > best.f1:
        best.f1 = f1
        best.epoch = epoch
        torch.save(model, 'best-conll.model')
    model.train(True)

print('Evaluating the best model on testb')
model = torch.load('best-conll.model')
model.train(False)
evaluate(model, data_testb, vocabs=vocabs, lbls=LABELS)
error_analysis(model, data_testb, vocabs=vocabs, verbose=False)

