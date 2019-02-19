import json
import os
import tensorboardX
from .bioes import convert_labels


def parse_conll2003(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line.startswith('-DOCSTART-'):
                continue
            if not line:
                if sentence:
                    yield sentence
                    sentence.clear()
            else:
                parts = line.split()
                sentence.append((parts[0], parts[-1]))
        if sentence:
            yield sentence


def read_conll2003(fname, normalize_digits=False, label_encoding='bioes'):

    for sentence in parse_conll2003(fname):
        wrds = [w for w,_ in sentence]
        if normalize_digits:
            wrds = [norm(w) for w in wrds]
        lbls = [l for _,l in sentence]
        lbls = list(convert_labels(lbls, label_encoding=label_encoding))

        yield {
            'words': wrds,
            'labels': lbls,
        }


def word_shape(word):

    if word[0].lower() != word[0]:
        if word[1:].lower() == word[1:]:
            return 'title'
        elif word[1:].upper() == word[1:]:
            return 'upper'
        else:
            return 'misc'
    else:
        if word[1:].lower() == word[1:]:
            return 'lower'
        else:
            return 'misc'


def norm(word):
    x =  re.sub(r'\d', '0', word)
    return x


def save_json(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


class SummaryWriter(tensorboardX.SummaryWriter):
    '''Allows transparently opening sub-writers (train, dev, etc), by indexing with the label.

        sw = SummaryWriter('traindir/summary')
        sw.add_scalar('loss', 0.1)  # this works as usual

        sw['dev'].add_scalar('loss', 0.085)  # this creates summary in "traindir/summary/dev"
    '''
    def __init__(self, log_dir):
        tensorboardX.SummaryWriter.__init__(self, log_dir)
        self._log_dir = log_dir
        self._subwriters = {}

    def __getitem__(self, label):
        if label not in self._subwriters:
            self._subwriters[label] = SummaryWriter(f'{self._log_dir}/{label}')
        return self._subwriters[label]

    def close(self):
        super().close()
        for subwriter in self._subwriters.values():
            subwriter.close()

    def flush(self):
        super().flush()
        for subwriter in self._subwriters.values():
            subwriter.flush()

    def add_scalar_metric(self, metrics :dict, global_step :int=None):
        '''Convenience method to deal with our dict metrics format'''
        for name, value in metrics.items():
            self.add_scalar(name, value, global_step=global_step)
