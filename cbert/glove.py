import numpy as np


class Glove:

    def __init__(self, words, vecs):
        assert words[0] == '<pad>'
        assert all(x==0. for x in vecs[0])
        assert len(vecs) == len(words)
        self._encoder = {
            x:y
            for x,y in zip(words, vecs)
        }
        self._unk = self._encoder['<unk>']

    def __getitem__(self, word):
        return self._encoder.get(word, self._unk)

    def __contains__(self, word):
        return word in self._encoder

    def __len__(self):
        return len(self._encoder)

    @property
    def dim(self):
        return self._unk.shape[0]

    @classmethod
    def load(cls, fname):

        words = []
        vecs = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                words.append(parts[0])
                vecs.append(np.array([float(x) for x in parts[1:]], dtype=np.float32))

        assert '<pad>' not in words
        assert '<unk>' in words

        words.insert(0, '<pad>')
        vecs.insert(0, np.zeros(shape=(len(vecs[0]),), dtype=np.float32))

        return cls(words, vecs)
