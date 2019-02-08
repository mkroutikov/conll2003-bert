import collections


INFTY = 10000.0

B = lambda l: f'B-{l}'
I = lambda l: f'I-{l}'
E = lambda l: f'E-{l}'
S = lambda l: f'S-{l}'


def decode_bioes_logits(T, *, logits_factory, labels):

    prev = {}  # records transitions
    alpha = collections.defaultdict(lambda: -INFTY)
    alpha[-1, 'O'] = 0.
    last_label_penalty = collections.defaultdict(float)

    for t in range(T):

        # transitions to B-label <== O, E-x, S-x
        for l in labels:
            logit = logits_factory(t, B(l))
            alpha[t, B(l)], prev[t, B(l)] = max(
                [(alpha[t-1, 'O'] + logit, 'O')] +  # <== O
                [(alpha[t-1, E(x)] + logit, E(x)) for x in labels] + # <== E-x
                [(alpha[t-1, S(x)] + logit, S(x)) for x in labels]   # <== S-x
            , key=lambda x: x[0])

        # transitions to S-label <== O, E-x, S-x
        for l in labels:
            logit = logits_factory(t, S(l))
            alpha[t, S(l)], prev[t, S(l)] = max(
                [(alpha[t-1, 'O'] + logit, 'O')] +  # <== O
                [(alpha[t-1, E(x)] + logit, E(x)) for x in labels] + # <== E-x
                [(alpha[t-1, S(x)] + logit, S(x)) for x in labels]   # <== S-x
            , key=lambda x: x[0])

        # transitions to I-label <== B-l, I-l
        for l in labels:
            logit = logits_factory(t, I(l))
            alpha[t, I(l)], prev[t, I(l)] = max([
                (alpha[t-1, B(l)] + logit, B(l)),
                (alpha[t-1, I(l)] + logit, I(l)),
            ], key=lambda x: x[0])

        # transitions to E-label <== B-l, I-l
        for l in labels:
            logit = logits_factory(t, E(l))
            alpha[t, E(l)], prev[t, E(l)] = max([
                (alpha[t-1, B(l)] + logit, B(l)),
                (alpha[t-1, I(l)] + logit, I(l)),
            ], key=lambda x: x[0])

        # transitions to O <== O, E-x, S-x
        logit = logits_factory(t, 'O')
        alpha[t, 'O'], prev[t, 'O'] = max(
            [(alpha[t-1, 'O'] + logit, 'O')] +  # <== O
            [(alpha[t-1, E(x)] + logit, E(x)) for x in labels] + # <== E-x
            [(alpha[t-1, S(x)] + logit, S(x)) for x in labels]   # <== S-x
        , key=lambda x: x[0])

    # Final transition to O (no logits) <== O, E-x, S-x
    alpha[T, 'O'], prev[T, 'O'] = max(
        [(alpha[T-1, 'O'], 'O')] +  # <== O
        [(alpha[T-1, E(x)], E(x)) for x in labels] + # <== E-x
        [(alpha[T-1, S(x)], S(x)) for x in labels]   # <== S-x
    , key=lambda x: x[0])

    best_value = alpha[T, 'O']
    best_label = prev[T, 'O']
    best_labels = []
    t = T
    while t > 0:
        best_labels.insert(0, best_label)
        best_label = prev[t-1, best_label]
        t -= 1

    return best_value, best_labels


def brute_force(T, logits_factory, labels):

    return brute_force_fixed_ends(0, T, 'O', 'O', logits_factory, labels)

    for l1 in xx:
        for l2 in yy:
            cost, path = brute_force_fixed_ends(T, logits_factory, labels, 0, l1, l2)

def labels_compatible(l1, l2):
    if l1[0] in ('O', 'E', 'S'):
        return l2[0] in ('O', 'B', 'S')
    elif l2[0] not in ('E', 'I'):
        return False
    else:  # (BI -- EI)
        return l1[2:] == l2[2:]

def brute_force_fixed_ends(start, end, l1, l2, logits_factory, labels):
    '''
    computes best path from start to (end-1) inclusive. Given that label at start-1 is l1, and label at end is l2

    Returns tuple: (score, path), where path is the array of length end-start
    '''
    if end == start:
        return (0, []) if labels_compatible(l1, l2) else (-INFTY, [0])

    middle = start + (end - start) // 2

    best_score = -INFTY
    best_path = ['O'] * (end-start)
    for l in ['O'] + [B(x) for x in labels] + [I(x) for x in labels] + [S(x) for x in labels] + [E(x) for x in labels]:
        logit = logits_factory(middle, l)
        score1, path1 = brute_force_fixed_ends(start, middle, l1, l, logits_factory, labels)
        score2, path2 = brute_force_fixed_ends(middle+1, end, l, l2, logits_factory, labels)
        if score1 + logit + score2 > best_score:
            best_score = score1 + logit + score2
            best_path = path1 + [l] + path2
    return best_score, best_path

