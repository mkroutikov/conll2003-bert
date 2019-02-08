
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
