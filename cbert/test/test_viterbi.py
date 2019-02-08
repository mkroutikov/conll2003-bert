from ..viterbi import decode_bioes_logits, INFTY, brute_force
import numpy as np
import random


B = lambda l: f'B-{l}'
I = lambda l: f'I-{l}'
E = lambda l: f'E-{l}'
S = lambda l: f'S-{l}'

labels = ['LOC', 'ORG', 'PER']

LABELS_ST  = ['O'] + ['B-'+x for x in labels] + ['S-'+x for x in labels]
LABELS_EN  = ['O'] + ['E-'+x for x in labels] + ['S-'+x for x in labels]
LABELS_ALL = ['O'] + ['B-'+x for x in labels] + ['I-'+x for x in labels] + ['E-'+x for x in labels] + ['S-'+x for x in labels]

vocab = {
    l: i
    for i,l in enumerate(LABELS_ALL)
}

def test_viterbi():
    for _ in range(1000):
        seqlen = random.randint(1, 10)
        logits = np.random.uniform(size=(seqlen, len(LABELS_ALL)))

        def logits_factory(t, l):
            return logits[t, vocab[l]]

        score1, path1 = brute_force(seqlen, logits_factory, labels)
        score2, path2 = decode_bioes_logits(seqlen, logits_factory=logits_factory, labels=labels)

        if abs(score1 - score2) > 1e-5:
            print(score1, path1)
            print(score2, path2)
            print()
            import pdb; pdb.set_trace()
            score1, path1 = brute_force(seqlen, logits_factory, labels)
            score2, path2 = decode_bioes_logits(seqlen, logits_factory=logits_factory, labels=labels)

