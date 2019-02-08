from ..vocab import Vocab


def test_smoke():
    vocab = Vocab()

    assert len(vocab) == 2
    assert vocab['<pad>'] == 0
    assert vocab['<unk>'] == 1

    vocab.update('The quick brown fox jumped over the lazy dog'.split())

    assert [vocab.encode(x) for x in 'The quick brown dog ate lazy fox'.split()] == [2, 3, 4, 10, 1, 9, 5]

    assert len(vocab) == 11

    assert vocab.values == [
        '<pad>',
        '<unk>',
        'The',
        'quick',
        'brown',
        'fox',
        'jumped',
        'over',
        'the',
        'lazy',
        'dog',
]