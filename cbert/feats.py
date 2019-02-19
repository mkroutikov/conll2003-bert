import os
import logging
from .vocab import Vocab


class Feats:
    '''Class is responsible for encoding tokens and labels into numbers'''

    def __init__(self, vocab=None):
        self.vocab = vocab

    def encode(self, *datasets):
        if self.vocab is None:
            self.vocab = build_vocabs(*datasets)

        out = []

        for dataset in datasets:
            out.append([])
            count = 0
            for item in dataset:
                data = encode(self.vocab, item)
                out[-1].append(data)
                count += 1

        return tuple(out) if len(out) > 1 else out[0]

    def save(self, datadir):
        os.makedirs(datadir, exist_ok=True)

        if self.vocab is None:
            raise RuntimeError('vocabularies were not built! can not save')

        self.vocab['words'].save(f'{datadir}/words.vocab')
        self.vocab['labels'].save(f'{datadir}/labels.vocab')

    @classmethod
    def load(cls, datadir):
        vocab = dict(
            words=Vocab.load(f'{datadir}/words.vocab'),
            labels=Vocab.load(f'{datadir}/labels.vocab'),
        )

        return cls(vocab)


def build_vocabs(*datasets):

    words_vocab  = Vocab()
    labels_vocab = Vocab()

    counter = 0
    for dataset in datasets:
        for item in dataset:
            words_vocab.update(item['words'])
            labels_vocab.update(item['labels'])
            counter += 1

    logging.info('Total records read: %s', counter)
    logging.info('Words vocab size: %s', len(words_vocab))
    logging.info('Labels vocab size: %s', len(labels_vocab))

    return dict(
        words =words_vocab,
        labels=labels_vocab,
    )


def encode(vocabs, item, show=False):
    '''Encodes one data item into a feature'''
    words_vocab = vocabs['words']

    out = dict(
        words = [words_vocab.encode(x) for x in item['words']],
    )

    if 'labels' in item:
        labels_vocab = vocabs['labels']
        out['labels'] = [labels_vocab.encode(x) for x in item['labels']]

    if show:
        display_words = ' '.join(words_vocab.decode(x) for x in out['words'])

        logging.info('Words: %s', display_words)

        if labels is not None:
            display_labels = ' '.join(labels_vocab.decode(x) for x in out['labels'])
            logging.info('Labels: %s', display_labels)

    return out
