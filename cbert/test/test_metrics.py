import torch
import pytest
import numpy as np
from ..metrics import TokenAndRecordAccuracy, F1Score, MetricSet
from ..metrics import TokenAndRecordAccuracyBert, F1ScoreBert
from ..vocab import Vocab
from ..metrics import Ema, Mean, massage_bert_logits_and_labels


def test_accuracy_metric():

    logits = torch.tensor([
        [
            [0., 0., 3., 2.],
            [0., 5., 0., 2.],
            [0., 1., 0., 7.],
        ]
    ])

    labels = torch.tensor([
        [2, 1, 1]
    ])

    a = TokenAndRecordAccuracy()
    a.append(logits, labels, loss=0.4)

    assert a.summary == {'racc': 0.0, 'tacc': pytest.approx(0.6666666666), 'loss': 0.4}


def test_accuracy_metric_bert():

    logits = torch.tensor([
        [
            [0., 5., 6., 7.],
            [0., 0., 3., 2.],
            [0., 5., 0., 2.],
            [0., 1., 0., 7.],
            [0., 5., 6., 7.],
        ]
    ])

    labels = torch.tensor([
        [1, 2, 1, 1, 1]
    ])

    a = TokenAndRecordAccuracyBert()
    a.append(logits, labels, loss=0.4)
    assert a.acc.value['tokens'] == 3
    assert a.acc.value['records'] == 1

    assert a.summary == {'racc': 0.0, 'tacc': pytest.approx(0.6666666666), 'loss': 0.4}


def test_f1_score():

    logits = torch.tensor([
        [
            [0., 0., 0., 3., 2., 0.],  # B-x
            [0., 0., 5., 0., 2., 6.],  # E-x
            [0., 0., 1., 0., 4., 7.],  # E-x
            [0., 0., 4., 0., 1., 0.],  # O
        ]
    ])

    labels = torch.tensor([
        [3, 5, 3, 5]
    ])

    labels_vocab = Vocab(['<pad>', '<unk>', 'O', 'B-x', 'I-x', 'E-x'])

    score = F1Score(labels_vocab=labels_vocab)
    score.append(logits, labels)

    assert score.summary == {
        'racc': 0.0,
        'f1': pytest.approx(0.666666),
        'prec': pytest.approx(1.),
        'recall': pytest.approx(0.5)
    }


def test_f1_score_bert():

    logits = torch.tensor([
        [
            [7., 100., 0., 3., 2.],
            [0., 0., 0., 3., 2.],   # B-x
            [0., 0., 1., 0., 3.],   # I-x
            [0., 0., 1., 0., 7.],   # I-x
            [0., 0., 3., 1., 1.],   # O
            [7., 100., 0., 3., 2.],
        ]
    ])

    labels = torch.tensor([
        [2, 3, 4, 4, 4, 2]
    ])

    labels_vocab = Vocab(['<pad>', '<unk>', 'O', 'B-x', 'I-x'])

    score = F1ScoreBert(labels_vocab=labels_vocab)
    score.append(logits, labels)

    assert score.summary == {'racc': pytest.approx(1.0), 'f1': pytest.approx(1.0), 'prec': pytest.approx(1.), 'recall': pytest.approx(1.0)}


def test_ema():

    ema = Ema(tau=0.1)
    ema += {'value': 1}
    ema += {'value': 2}

    assert ema.value == {'value': 1.1}


def test_mean():

    mean = Mean()
    mean += {'value': 1}
    mean += {'value': 2, 'other_value': 5}

    assert mean.value == {'value': 1.5, 'other_value': 5}

    mean.reset().update({'value': 1} for _ in range(10))
    assert mean.value == {'value': 1}
    assert mean.accumulator('value') == 10
    assert mean.count('value') == 10

    assert Mean({'value': x} for x in range(5)).value == {'value': 2.0}


def test_metric_manager():

    logits = torch.tensor([
        [
            [0., 0., 0., 3., 2.],
            [0., 0., 5., 0., 2.],
            [0., 0., 1., 0., 7.],
            [0., 0., 1., 0., 7.],
        ]
    ])

    labels = torch.tensor([
        [2, 3, 2, 4]
    ])

    labels_vocab = Vocab(['<pad>', '<unk>', 'O', 'B-x', 'I-x'])

    metric = MetricSet({
        'acc': TokenAndRecordAccuracy(),
        'entity': F1Score(labels_vocab=labels_vocab),
    })

    metric.append(logits, labels)

    assert metric.summary == {
        'acc.racc': 0.0,
        'acc.tacc': pytest.approx(0.25),
        'entity.f1': 0.0,
        'entity.prec': 0.0,
        'entity.racc': 0.0,
        'entity.recall': 0.0
    }


def test_massage_bert():

    logits = torch.zeros(3, 5, 10, dtype=torch.float32)
    labels = torch.tensor([
        [1, 2, 3, 4, 5],
        [5, 6, 7, 1, 0],
        [8, 9, 1, 0, 0],
    ])

    out, target = massage_bert_logits_and_labels(logits, labels)
    assert out.size(0) == 3
    assert out.size(1) == 3
    assert out.size(2) == 10
    assert target.size(0) == 3
    assert target.size(1) == 3

    model = torch.tensor([
        [2, 3, 4],
        [6, 7, 0],
        [9, 0, 0],
    ])
    assert (target != model).long().sum() == 0
