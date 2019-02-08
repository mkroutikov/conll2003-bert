class Vocab:
    ''' Simple vocabulary. Does not support newlines or items with newlines in it '''

    def __init__(self, values=None):
        if values is None:
            values = ['<pad>', '<unk>']
        else:
            assert values[0] == '<pad>'
            assert values[1] == '<unk>'

        self._decoder = {i:val for i,val in enumerate(values)}
        self._encoder = {val:i for i,val in enumerate(values)}

    def update(self, items):
        for item in items:
            if item not in self._encoder:
                idx = len(self)
                self._encoder[item] = idx
                self._decoder[idx] = item

    def __len__(self):
        return len(self._encoder)

    def __getitem__(self, val):
        return self.encode(val)

    def __contains__(self, val):
        return val in self._encoder

    def encode(self, val):
        return self._encoder.get(val, 1)  # 1 is <unk>

    def decode(self, idx):
        return self._decoder[idx]

    @property
    def values(self):
        return [self.decode(i) for i in range(len(self))]

    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.values) + '\n')

    @classmethod
    def load(cls, fname):
        with open(fname, 'r', encoding='utf-8') as f:
            return cls(f.read().rstrip('\n').split('\n'))

    @classmethod
    def from_tokens(cls, tokens):
        assert '<pad>' not in tokens
        assert '<unk>' not in tokens
        return cls(['<pad>', '<unk>'] + list(tokens))

