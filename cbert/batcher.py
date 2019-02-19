import torch
import torch.utils.data


class BatcherBase:
    ''' A thin wrapper around DataLoader, uses abstract collation '''

    def __init__(self, dataset, batch_size, shuffle=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._device = None

    def __iter__(self):
        return iter(torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            collate_fn=self._collate
        ))

    def to(self, device):
        self._device = device
        return self

    def _collate(self, batch):
        raise NotImplementedError()


class Batcher(BatcherBase):
    '''Batcher for variable-length sequences'''

    def __init__(self, dataset, batch_size, shuffle=False, max_seqlen=100000, sort=False):
        BatcherBase.__init__(self, dataset, batch_size, shuffle)
        self._max_seqlen = max_seqlen
        self._sort = sort

    def _collate(self, batch):
        if self._sort:
            batch = sorted(batch, key=lambda x: -len(x['words']))  # sort by descending len

        maxlen = max(len(x['words']) for x in batch)
        if maxlen > self._max_seqlen:
            maxlen = self._max_seqlen

        batch_size = len(batch)

        words  = torch.zeros(batch_size, maxlen, dtype=torch.int64)  # zero is our padding index!
        labels = torch.zeros(batch_size, maxlen, dtype=torch.int64)
        for i,item in enumerate(batch):
            x = torch.tensor(item['words'], dtype=torch.int64)[:maxlen]
            y = torch.tensor(item['labels'], dtype=torch.int64)[:maxlen]
            assert x.size(0) == y.size(0)
            words[i,:x.size(0)] = x
            labels[i,:y.size(0)] = y

        if self._device is not None:
            words = words.to(self._device)
            labels = labels.to(self._device)

        return words, labels
