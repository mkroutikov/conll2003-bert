import collections
import numbers
import torch
import torch.nn.functional as F
from types import SimpleNamespace as nm
from .bioes import entities_jie_bioes
from .viterbi import decode_bioes_logits, INFTY


EPSILON = 1.e-8


def token_and_record_accuracy(logits, labels):
    '''Computes accuracy metric from logits and true labels.

    Takes care not to count padding for variable-length records.

    Input:
        logits - float tensor [B, S, L]
        labels - integer tensor [B, S]

    Where:
        B - batch size (> 0)
        S - max sequence length (> 0). Labels are padded with zeroes.
        L - size of label vocabulary

    Returns:
        a dict with the following keys:

            records - number of records (aka batch size)
            correct_records - how many records were correct (a
                record is correct when all its tokens were predicted correctly)
            tokens - number of tokens in this batch
            correct_tokens - how many tokens were correctly predicted in this batch
    '''
    assert logits.size(0) == labels.size(0)

    batch_size = logits.size(0)

    mask = (labels > 0).long()
    seqlen = mask.sum(dim=1)

    pred = torch.argmax(logits, dim=2)

    correct_toks = ((pred==labels).long() * mask).sum(dim=1)
    numtokens = seqlen.sum().item()

    return dict(
        records         = batch_size,
        correct_records = (correct_toks==seqlen).long().sum().item(),
        tokens          = numtokens,
        correct_tokens  = correct_toks.sum().item(),
    )


def massage_bert_logits_and_labels(logits, labels):
    '''Fixes BERT logits and labels.
    BERT uses leading and trailing dummy tokens ([CLS] and [SEP]), and therefore
    its logits and labels can not be directly used to compute performance metric.
    This function removes dummy tokens.
    '''
    assert logits.size(0) == labels.size(0)
    assert logits.size(1) == labels.size(1)

    # skip leading [CLS] and trailing [SEP] tokens
    mask = (labels > 0).long()[:,2:]
    logits = logits[:,1:-1,:]
    labels = labels[:,1:-1] * mask

    return logits, labels


def entity_scores(logits, labels, labels_vocab, entity_decoder='fast', bioes_labels=None):
    '''Computes entity performance metric.

    Input:
        logits - a float tensor of shape [B, S, L]
        labels - an int tensor of shape [B, S], labels are zero-padded
        labels_vocab - Vocab object to decode labels
        entity_decoder = one of 'fast' or 'viterbi'. Dictates how to decode logits:
            'fast' - use argmax on logits and then use heuristic to resolve invalid label sequences
            'viterbi' - use Viterbi decoder to find optimal valid label sequence
        bioes_labels - set of labels (before I-,B-,E-,S- prefixes are added).
            Only needed for Viterbi decoder.

    Returns:
        a dict with the following keys:

            tp - number of true positive entities
            fp - number of false positive entities
            fn - number of false negative entities
            records - number of records processed (aka batch size)
            correct_records - number of records where true and predicted 
                entities matched exactly
    '''
    if entity_decoder not in ('fast', 'viterbi'):
        raise ValueError('Invalid value for "entity_decoder". Expect "fast" or "viterbi". Got: ' + entity_decoder)

    if entity_decoder == 'viterbi' and bioes_labels is None:
        raise ValueError('Parameter "bioes_labels" is required when "entity_decoder" is "viterbi"')

    batch_size = logits.size(0)

    pred = torch.argmax(logits, dim=2)
    mask = (labels > 0).long()
    seqlen = mask.sum(dim=1)

    def logits_factory_factory(logits, labels_vocab):

        def logits_factory(t, label):
            if label not in labels_vocab:
                return -INFTY
            return logits[t, labels_vocab.encode(label)]
        return logits_factory

    tp = 0
    fp = 0
    fn = 0
    records = 0
    correct_records = 0
    for i in range(batch_size):
        true_labels = labels[i, :seqlen[i]].tolist()
        true_labels = [labels_vocab.decode(i) for i in true_labels]
        true_entities = set(entities_jie_bioes(true_labels))

        if entity_decoder == 'viterbi':
            _, pred_labels = decode_bioes_logits(
                seqlen[i].item(),
                logits_factory=logits_factory_factory(logits[i], labels_vocab=labels_vocab),
                labels=bioes_labels
            )
        else:
            pred_labels = pred[i,:seqlen[i]].tolist()
            pred_labels = [labels_vocab.decode(i) for i in pred_labels]
        pred_entities = set(entities_jie_bioes(pred_labels))

        tpx = len(pred_entities & true_entities)
        fpx = len(pred_entities - true_entities)
        fnx = len(true_entities - pred_entities)

        if fpx == 0 and fnx == 0:
            correct_records += 1
        records += 1
        tp += tpx
        fn += fnx
        fp += fpx

    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        records=records,
        correct_records=correct_records,
    )


def get_bioes_labels(labels):
    '''Computes set of core labels from their BIOES expansion'''
    labels = list(x for x in labels if x not in ('O', '<pad>', '<unk>'))
    assert all(x[:2] in ('B-', 'I-', 'E-', 'S-') for x in labels)
    return set(x[2:] for x in labels)


class Mean:
    '''Mean aggregator'''
    def __init__(self, iterable=None):
        self._acc = collections.defaultdict(float)
        self._num = collections.defaultdict(int)

        if iterable:
            self.update(iterable)

    def reset(self):
        self._acc.clear()
        self._num.clear()
        return self

    def update(self, iterable):
        for val in iterable:
            self += val

    def __iadd__(self, other):
        for key, value in other.items():
            self._acc[key] += value
            self._num[key] += 1
        return self

    @property
    def value(self):
        return  {
            key: val / self._num[key]
            for key, val in self._acc.items()
        }

    def accumulator(self, name):
        return self._acc[name]

    def count(self, name):
        return self._num[name]


class Ema:
    '''EMA aggregator'''
    def __init__(self, tau=0.1):
        self._value = collections.defaultdict(float)
        self.tau = tau

    def reset(self):
        self._value = None

    def __iadd__(self, other):
        for key, val in other.items():
            if key not in self._value:
                self._value[key] = val
            else:
                self._value[key] += (val - self._value[key]) * self.tau
        return self

    @property
    def value(self):
        return dict(self._value)


class Metric:
    def __init__(self, acc=None):
        if acc is None:
            acc = Mean()  # by default, compute mean statistics
        self.acc = acc

    def reset(self):
        self.acc.reset()
        return self

    def append(self, output, target, **losses):
        raise NotImplementedError()

    def update(self, iterable):
        for output, target in iterable:
            self.append(output, target)

    @property
    def summary(self):
        return self.acc.value

    def __repr__(self):
        return repr(self.summary)


class TokenAndRecordAccuracy(Metric):

    def append(self, output, target, **losses):
        self.acc += token_and_record_accuracy(output, target)
        self.acc += losses
        return self

    @property
    def summary(self):
        summ = self.acc.value
        tokens = summ.pop('tokens')
        correct_tokens = summ.pop('correct_tokens')
        records = summ.pop('records')
        correct_records = summ.pop('correct_records')

        summ['racc'] = correct_records / (records + EPSILON)
        summ['tacc'] = correct_tokens / (tokens + EPSILON)

        return summ


class TokenAndRecordAccuracyBert(TokenAndRecordAccuracy):

    def append(self, output, target, **losses):
        output, target = massage_bert_logits_and_labels(output, target)
        return super().append(output, target, **losses)


class F1Score(Metric):

    def __init__(self, labels_vocab, entity_decoder='fast', acc=None):
        Metric.__init__(self, acc)
        if entity_decoder not in ('fast', 'viterbi'):
            raise ValueError('Invalid value for "entity_decoder", accept only "fast" and "viterbi": ' + entity_decoder)
        self._labels_vocab = labels_vocab
        self._entity_decoder = entity_decoder
        self._bioes_labels = get_bioes_labels(labels_vocab.values)

    def append(self, output, target, **losses):
        self.acc += entity_scores(output, target,
            labels_vocab=self._labels_vocab, entity_decoder=self._entity_decoder, bioes_labels=self._bioes_labels)
        self.acc += losses
        return self

    @property
    def summary(self):
        summ = self.acc.value
        tp, fp, fn = summ.pop('tp'), summ.pop('fp'), summ.pop('fn')
        records = summ.pop('records')
        correct_records = summ.pop('correct_records')

        prec = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)
        f1 = 2 * prec * recall / (prec + recall + EPSILON)

        summ['prec'] = prec
        summ['recall'] = recall
        summ['f1'] = f1
        summ['racc'] = correct_records / (records + EPSILON)

        return summ


class F1ScoreBert(F1Score):

    def append(self, output, target, **losses):
        output, target = massage_bert_logits_and_labels(output, target)
        return super().append(output, target, **losses)


class CrossEntropyLoss(Metric):

    def append(self, output, target, **losses):
        output = output.transpose(1, 2)
        loss = F.cross_entropy(output, target, ignore_index=0, reduction='mean')
        self.acc += {'loss': loss.item()}
        return self


class MetricSet(Metric):

    def __init__(self, *configs):
        self._configs = configs

    def reset(self):
        for config in self._configs:
            for metric in config.values():
                metric.reset()
        return self

    def append(self, output, target, **losses):
        for config in self._configs:
            for metric in config.values():
                metric.append(output, target, **losses)

    @property
    def summary(self):
        out = {}
        for config in self._configs:
            for key, m in config.items():
                for subkey, value in m.summary.items():
                    out[key+'.'+subkey] = value
        return out
