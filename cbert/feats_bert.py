import os
import logging
from .vocab import Vocab
from .util import read_conll2003
from pytorch_pretrained_bert.tokenization import BertTokenizer
import itertools


class FeatsBert:
    '''Class is responsible for encoding tokens and labels into numbers'''

    def __init__(self, bert_model, show=0):
        self._bert_model = bert_model
        self.vocab = None

    def encode(self, *datasets):
        if self.vocab is None:
            self.vocab = build_vocabs(*datasets, bert_model=self._bert_model)

        out = []
        for dataset in datasets:
            out.append([])
            count = 0
            for item in dataset:
                data = encode(self.vocab, item)
                out[-1].append(data)
                count += 1

        return tuple(out)

    def save(self, datadir):
        os.makedirs(datadir, exist_ok=True)
        if self.vocab is None:
            raise RuntimeError('vocabularies were not built! can not save')

        self.vocab['labels'].save(f'{datadir}/labels.vocab')

    @classmethod
    def load(cls, datadir, bert_model):
        do_lower_case = bert_model.endswith('-uncased')
        vocab = dict(
            subwords=BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case),
            labels=Vocab.load(basedir + '/labels.vocab'),
        )

        encoder = cls(bert_model=bert_model)
        encoder.vocab = vocab

        return encoder


def build_vocabs(*datasets, bert_model='bert-base-multilingual-cased'):
    labels_vocab = Vocab()

    counter = 0
    for dataset in datasets:
        for item in dataset:
            labels_vocab.update(item['labels'])
            counter += 1

    logging.info('Total records read: %s', counter)
    logging.info('Labels vocab size: %s', len(labels_vocab))

    do_lower_case = bert_model.endswith('-uncased')
    return dict(
        subwords=BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case),
        labels=labels_vocab,
    )


def expand_labels(label, length):
    if length == 1 or label[0] in ('O', 'I'):
        return [label] * length
    elif label[0] == 'B':
        return [label] + ['I-'+label[2:]] * (length-1)
    elif label[0] == 'E':
        return ['I-'+label[2:]] * (length-1) + [label]
    elif label[0] == 'S':
        return ['B-'+label[2:]] + ['I-'+label[2:]] * (length-2) + ['E-'+label[2:]]
    else:
        assert False (label, length)


def encode(vocabs, item, show=False):
    words_vocab = vocabs['subwords']

    cls_id, sep_id = words_vocab.convert_tokens_to_ids(['[CLS]', '[SEP]'])

    words = [words_vocab.tokenize(x) for x in item['words']]

    if 'labels' in item:
        assert len(item['labels']) == len(words)
        labels = [expand_labels(l, len(w)) for l,w in zip(item['labels'], words)]
        labels = list(itertools.chain.from_iterable(labels))  # flatten

    words = list(itertools.chain.from_iterable(words))  # flatten

    words = [cls_id] + words_vocab.convert_tokens_to_ids(words) + [sep_id]
    out = {
        'words': words
    }

    if 'labels' in item:
        labels = ['O'] + labels + ['O']
        assert len(labels) == len(words)
        labels_vocab = vocabs['labels']
        out['labels'] = [labels_vocab.encode(x) for x in labels]

    if show:
        display_words = ' '.join(words_vocab.convert_ids_to_tokens(out['words']))

        logging.info('Words: %s', display_words)

        if labels is not None:
            labels_vocab = vocabs['labels']
            display_labels = ' '.join(labels_vocab.decode(x) for x in out['labels'])
            logging.info('Labels: %s', display_labels)

    return out
