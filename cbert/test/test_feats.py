from ..feats import Feats


def test_smoke():

    train_dataset = [
        {
            'words' : ['Once', 'upon', 'a', 'time', 'in', 'Boston'],
            'labels': ['O',    'O',    'O', 'O',    'O',  'S-LOC'],
        },
        {
            'words' : ['Mr.', 'Boss', 'opened', 'a', 'meeting'],
            'labels': ['O',   'S-PER', 'O',     'O', 'O'],
        },
    ]

    test_dataset = [
        {
            'words' : ['Here', 'in', 'New', 'York', 'City'],
            'labels': ['O',    'O',  'B-LOC', 'I-LOC', 'E-LOC'],
        },
    ]

    test_dataset

    feats = Feats()
    train, test = feats.encode(train_dataset, test_dataset)

    assert feats.vocab['words'].values == [
        '<pad>',
        '<unk>',
        'Once',
        'upon',
        'a',
        'time',
        'in',
        'Boston',
        'Mr.',
        'Boss',
        'opened',
        'meeting',
        'Here',
        'New',
        'York',
        'City'
    ]

    assert feats.vocab['labels'].values == ['<pad>', '<unk>', 'O', 'S-LOC', 'S-PER', 'B-LOC', 'I-LOC', 'E-LOC']
    assert train == [
        {'labels': [2, 2, 2, 2, 2, 3], 'words': [2, 3, 4, 5, 6, 7]},
        {'labels': [2, 4, 2, 2, 2], 'words': [8, 9, 10, 4, 11]},
    ]
    assert test == [
        {'words': [12, 6, 13, 14, 15], 'labels': [2, 2, 5, 6, 7]},
    ]