import numpy as np
from .vocab import Vocab
import logging
import itertools
import re
from .bioes import convert_labels
from .util import parse_conll2003, norm


def build_vocabs(fnames, label_encoding='bioes', normalize_digits=False):

    words_vocab  = Vocab()
    labels_vocab = Vocab()

    counter = 0
    for fname in fnames:
        logging.info('Reading data from %s', fname)
        for sentence in parse_conll2003(fname):
            wrds = [w for w,_ in sentence]
            if normalize_digits:
                wrds = [norm(w) for w in wrds]
            lbls = [l for _,l in sentence]
            lbls = list(convert_labels(lbls, label_encoding=label_encoding))

            words_vocab.update(x for x in wrds)
            labels_vocab.update(lbls)
            counter += 1

    logging.info('Total records read: %s', counter)
    logging.info('Words vocab size: %s', len(words_vocab))
    logging.info('Labels vocab size: %s', len(labels_vocab))

    return dict(
        words =words_vocab,
        labels=labels_vocab,
    )


def load_vocabs(basedir):
    return dict(
        words =Vocab.load(basedir + '/words.vocab'),
        labels=Vocab.load(basedir + '/labels.vocab'),
    )


def save_vocabs(vocabs, basedir):
    vocabs['words'].save(basedir + '/words.vocab')
    vocabs['labels'].save(basedir + '/labels.vocab')


def encode(vocabs, words, labels=None, show=False):
    words_vocab = vocabs['words']

    out = dict(
        words = [words_vocab.encode(x) for x in words],
    )

    if labels is not None:
        labels_vocab = vocabs['labels']
        out['labels'] = [labels_vocab.encode(x) for x in labels]

    if show:
        display_words = ' '.join(words_vocab.decode(x) for x in out['words'])

        logging.info('Words: %s', display_words)

        if labels is not None:
            display_labels = ' '.join(labels_vocab.decode(x) for x in out['labels'])
            logging.info('Labels: %s', display_labels)

    return out


def preprocess(fname, *, vocabs, show_first=0, label_encoding='bioes', normalize_digits=False):

    dataset = []

    count = 0
    for sentence in parse_conll2003(fname):
        wrds = [w for w,_ in sentence]
        if normalize_digits:
            wrds = [norm(w) for w in wrds]
        lbls = [l for _,l in sentence]
        lbls = list(convert_labels(lbls, label_encoding=label_encoding))

        data = encode(vocabs, wrds, lbls, show=count<show_first)
        dataset.append(data)
        count += 1

    return dataset


if __name__ == '__main__':
    import argparse
    import pickle
    import os

    parser = argparse.ArgumentParser(description='Preprocesses CoNLL data into dense numpy tensors')
    parser.add_argument('--label_encoding', default='bioes', help='label encoding scheme to use,  one of: "bio", "bioes" [%(default)s]')
    parser.add_argument('--normalize_digits', action='store_true', default=False, help='If set, replace all digits with zeroes [%(default)s]')
    parser.add_argument('--outdir', required=True, help='output directory (will be created if does not exist)')
    parser.add_argument('file', nargs='+', help='files to process. Note that vocabularies are built from all files.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.outdir, exist_ok=True)

    if all(os.path.isfile(f'{args.outdir}/{name}.vocab') for name in ['words', 'labels']):
        logging.info('Vocabularies found, will not recompute')
        vocabs = load_vocabs(args.outdir)
    else:
        logging.info('Vocabularies not found, building...')
        vocabs = build_vocabs(args.file, label_encoding=args.label_encoding, normalize_digits=args.normalize_digits)
        save_vocabs(vocabs, args.outdir)
        logging.info('Vocabularies saved')

    for fname in args.file:
        data = preprocess(
            fname,
            vocabs=vocabs,
            show_first=1,
            label_encoding=args.label_encoding,
            normalize_digits=args.normalize_digits,
        )
        base = os.path.basename(fname)

        outname = f'{args.outdir}/{base}.pickle'
        with open(outname, 'wb') as f:
            pickle.dump(data, f)
        logging.info('Saved %s', outname)
