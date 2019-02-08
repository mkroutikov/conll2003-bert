import itertools
from ..bioes import entities, Entity, entities_to_bio, entities_to_bioes


def test_smoke():

    labels = ['O', 'B-mike', 'B-something', 'I-blah', 'I-blah', 'E-blah']
    ents = list(entities(labels))
    assert ents == [
        Entity('mike', 1, 2),
        Entity('something', 2, 3),
        Entity('blah', 3, 6),
    ]

    bio_labels = list(itertools.islice(entities_to_bio(ents), 6))
    assert bio_labels == ['O', 'B-mike', 'B-something', 'B-blah', 'I-blah', 'I-blah']

    bioes_labels = list(itertools.islice(entities_to_bioes(ents), 6))
    assert bioes_labels == ['O', 'S-mike', 'S-something', 'B-blah', 'I-blah', 'E-blah']
