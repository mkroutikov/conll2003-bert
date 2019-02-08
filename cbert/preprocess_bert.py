import numpy as np
from .vocab import Vocab
import logging
import itertools
import re
from .bioes import convert_labels
from .util import parse_conll2003
from pytorch_pretrained_bert.tokenization import BertTokenizer


def build_vocabs(fnames, label_encoding='bioes', normalize_digits=False, bert_model='bert-base-multilingual-cased'):

    words_vocab  = Vocab()
    labels_vocab = Vocab()

    counter = 0
    for fname in fnames:
        logging.info('Reading data from %s', fname)
        for sentence in parse_conll2003(fname):
            lbls = [l for _,l in sentence]
            lbls = list(convert_labels(lbls, label_encoding=label_encoding))

            labels_vocab.update(lbls)
            counter += 1

    logging.info('Total records read: %s', counter)
    logging.info('Labels vocab size: %s', len(labels_vocab))

    do_lower_case = bert_model.endswith('-uncased')
    return dict(
        subwords=BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case),
        labels=labels_vocab,
    )


def load_vocabs(basedir, bert_model):
    do_lower_case = bert_model.endswith('-uncased')
    return dict(
        subwords=BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case),
        labels=Vocab.load(basedir + '/labels.vocab'),
    )


def save_vocabs(vocabs, basedir):
    vocabs['labels'].save(basedir + '/labels.vocab')


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


def encode(vocabs, words, labels=None, show=False):
    words_vocab = vocabs['subwords']

    cls_id, sep_id = words_vocab.convert_tokens_to_ids(['[CLS]', '[SEP]'])

    words = [words_vocab.tokenize(x) for x in words]

    if labels is not None:
        assert len(labels) == len(words)
        labels = [expand_labels(l, len(w)) for l,w in zip(labels, words)]
        labels = list(itertools.chain.from_iterable(labels))  # flatten

    words = list(itertools.chain.from_iterable(words))  # flatten

    words = [cls_id] + words_vocab.convert_tokens_to_ids(words) + [sep_id]
    out = {
        'words': words
    }

    if labels is not None:
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
    parser.add_argument('--bert_model', default='bert-base-multilingual-cased', help='Name of the BERT pretrained tokenizer [%(default)s]')
    parser.add_argument('--outdir', required=True, help='output directory (will be created if does not exist)')
    parser.add_argument('file', nargs='+', help='files to process. Note that vocabularies are built from all files.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.outdir, exist_ok=True)

    if all(os.path.isfile(f'{args.outdir}/{name}.vocab') for name in ['words', 'labels']):
        logging.info('Vocabularies found, will not recompute')
        vocabs = load_vocabs(args.outdir, args.bert_model)
    else:
        logging.info('Vocabularies not found, building...')
        vocabs = build_vocabs(
            args.file,
            label_encoding=args.label_encoding,
            normalize_digits=args.normalize_digits,
            bert_model=args.bert_model
        )
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
