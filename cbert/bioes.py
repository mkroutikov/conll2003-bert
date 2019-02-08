'''
Utilities to:
* fix broken BIO encoding
* convert to BIOES
* compute Precision/Recall/F1 based on BIO and BIOS
'''
from types import SimpleNamespace
import collections
import itertools


Entity = collections.namedtuple('Entity', ['label', 'start', 'end'])


def entities(labels):
    pending = None
    for i,l in enumerate(labels):
        if l[:2] == 'B-':
            if pending:
                yield Entity(label=pending.label, start=pending.start, end=i)
            pending = SimpleNamespace(start=i, label=l[2:])
        elif l[:2] == 'I-':
            if pending is None:  # fixit by implying B-
                pending = SimpleNamespace(start=i, label=l[2:])
            elif pending.label != l[2:]:
                yield Entity(label=pending.label, start=pending.start, end=i)
                pending = SimpleNamespace(start=i, label=l[2:])
        elif l == 'O':
            if pending is not None:
                yield Entity(label=pending.label, start=pending.start, end=i)
                pending = None
        elif l[:2] == 'E-':
            if pending is None:  # fixit by implying S-
                yield Entity(label=l[2:], start=i, end=i+1)
            elif pending.label != l[2:]:
                yield Entity(label=pending.label, start=pending.start, end=i)
                pending = None
                yield Entity(label=l[2:], start=i, end=i+1)
            else:
                yield Entity(label=pending.label, start=pending.start, end=i+1)
                pending=None
        elif l[:2] == 'S-':
            if pending is not None:
                yield Entity(label=pending.label, start=pending.start, end=i)
                pending = None
            yield Entity(label=l[2:], start=i, end=i+1)
        else:
            raise RuntimeError(f'Unrecognized label: {l}')

    if pending:
        yield Entity(label=pending.label, start=pending.start, end=len(labels))


def entities_to_bio(entities):
    ''' converts entities to BIO label encoding scheme '''

    offset = 0
    for e in entities:
        assert offset <= e.start
        assert e.end > e.start
        if offset < e.start:
            yield from itertools.repeat('O', e.start-offset)
        yield 'B-' + e.label
        if e.end > e.start + 1:
            yield from itertools.repeat('I-'+e.label, e.end-e.start-1)
        offset = e.end

    yield from itertools.repeat('O')


def entities_to_bioes(entities):
    ''' converts entities to BIO label encoding scheme '''

    offset = 0
    for e in entities:
        assert offset <= e.start
        assert e.end > e.start
        if offset < e.start:
            yield from itertools.repeat('O', e.start-offset)
        if e.end > e.start + 1:
            yield 'B-' + e.label
            yield from itertools.repeat('I-'+e.label, e.end-e.start-2)
            yield 'E-' + e.label
        else:
            yield 'S-' + e.label
        offset = e.end

    yield from itertools.repeat('O')


def subtokenize_bio(label, count):
    assert count > 0

    if label[:2] == 'B-':
        return [label] + ['I-'+label[2:]] * (count-1)
    else:
        return [label] * count

def subtokenize_bioes(label, count):
    assert count > 0

    if label[:2] == 'B-':
        return [label] + ['I-'+label[2:]] * (count-1)
    elif label[:2] == 'E-':
        return ['I-'+label[2:]] * (count-1) + [label]
    elif label[:2] == 'S-':
        if count == 1:
            return [label]
        else:
            return ['B-'+label[2:]] + ['I-'+label[2:]]*(count-2) + ['E-'+label[2:]]
    else:
        return [label] * count


POLICIES = {
    'B' : '0123',
    'I1': '01',
    'I2': '012',
    'O' : '012',
    'E1': '01',
    'E2': '01234',
    'S' : '0123',
    'F' : '01',
}

STANDARD_POLICY = {
    'B' : '3',
    'I1': '1',
    'I2': '0',
    'O' : '1',
    'E1': '0',
    'E2': '1',
    'S' : '0',
    'F' : '1',
}

MIKES_POLICY = {
    'B' : '0',
    'I1': '0',
    'I2': '0',
    'O' : '0',
    'E1': '0',
    'E2': '0',
    'S' : '0',
    'F' : '0',
}

def entities_policy(labels, policy=STANDARD_POLICY):
    pending = None
    for i,l in enumerate(labels):
        if l[:2] == 'B-':
            if pending:
                # need resolution. Choices are:
                # 0 - emit old as-it and start new
                # 1 - emit old as-is and do not start new
                # 2 - extend old and emit, do not start new
                # 3 - extend old and do not start new
                action = policy['B']
                if action == '0':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
                elif action == '1':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                elif action == '2':
                    yield Entity(pending.label, pending.start, i+1)
                    pending = None
                elif action == '3':
                    pending.end = i + 1
                else:
                    assert False
            else:
                pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
        elif l[:2] == 'I-':
            if pending is None:  # fixit by implying B-
                # need resolution. Choices are:
                # 0 - start a new sequence
                # 1 - no not start new sequence
                action = policy['I1']
                if action == '0':
                    pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
                elif action == '1':
                    pass
                else:
                    assert False
            elif pending.label != l[2:]:
                # need resolution
                # 0 - emit old as-is, and start new
                # 1 - emit old as-is, and do not start new
                # 2 - extend old, and do not start new
                action = policy['I1']
                if action == '0':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
                elif action == '1':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                elif action == '2':
                    pending.end = i + 1
                else:
                    assert False
            else:
                pending.end += 1
        elif l == 'O':
            if pending is not None:
                # need resolution. Choices are:
                # 0 - emit old as-is
                # 1 - discard old
                # 2 - extend old
                action = policy['O']
                if action == '0':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                elif action == '1':
                    pending = None
                elif action == '2':
                    pending.end = i + 1
                else:
                    assert False
        elif l[:2] == 'E-':
            if pending is None:  # fixit by implying S-
                # need resolution. Choices are:
                # 0 - emit one
                # 1 - do nothing
                # 2 - start new one
                action = policy['E1']
                if action == '0':
                    yield Entity(label=l[2:], start=i, end=i+1)
                elif action == '1':
                    pass
                elif action == '2':
                    pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
                else:
                    assert False
            elif pending.label != l[2:]:
                # need resolution. Choices are:
                # 0 - emit old and emit single
                # 1 - ignore old and emit single
                # 2 - emit old and ignore current
                # 3 - extend old and emit
                # 4 - extend old and continue
                action = policy['E1']
                if action == '0':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                    yield Entity(label=l[2:], start=i, end=i+1)
                elif action == '1':
                    pending = None
                    yield Entity(label=l[2:], start=i, end=i+1)
                elif action == '2':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                elif action == '3':
                    pending.end = i + 1
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                elif action == '4':
                    pending.end = i + 1
                else:
                    assert False
            else:
                pending.end += 1
                yield Entity(pending.label, pending.start, pending.end)
                pending = None
        elif l[:2] == 'S-':
            if pending is not None:
                # 0 - emit old and emit single
                # 1 - emit old and ignore current
                # 2 - extend old and emit
                # 3 - extend old and continue
                action = policy['E1']
                if action == '0':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                    yield Entity(label=l[2:], start=i, end=i+1)
                elif action == '1':
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                elif action == '2':
                    pending.end = i + 1
                    yield Entity(pending.label, pending.start, pending.end)
                    pending = None
                elif action == '3':
                    pending.end = i + 1
                else:
                    assert False
            else:
                yield Entity(label=l[2:], start=i, end=i+1)
        else:
            raise RuntimeError(f'Unrecognized label: {l}')

    if pending:
        # 0 - emit old
        action = policy['F']
        if action == '0':
            yield Entity(pending.label, pending.start, pending.end)
        elif action == '1':
            pass


def entities_jie_bioes(labels):
    pending = None

    for i,l in enumerate(labels):
        if l[:2] == 'B-':
            if pending:
                yield Entity(pending.label, pending.start, i)
                pending = None
            pending = SimpleNamespace(start=i, end=i+1, label=l[2:])
        elif l[:2] == 'S-':
            if pending is not None:
                yield Entity(pending.label, pending.start, i)
                pending = None
            yield Entity(label=l[2:], start=i, end=i+1)
        elif l[:2] == 'E-':
            if pending is not None:  # jie does not check if B- uses the same label!
                yield Entity(pending.label, pending.start, i + 1)
                pending = None

    if pending:
        yield Entity(pending.label, pending.start, pending.end)


def convert_labels(labels, *, label_encoding='bioes'):
    labels = list(labels)  # in case it was a generator
    ent = entities(labels)

    if label_encoding == 'bio':
        converter = entities_to_bio
    elif label_encoding == 'bioes':
        converter = entities_to_bioes
    else:
        raise ValueError(f'Unknown label encoding: {label_encoding}')

    return list(itertools.islice(converter(ent), len(labels)))


