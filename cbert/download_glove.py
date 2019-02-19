from pytorch_pretrained_bert.file_utils import cached_path
import os
import shutil
import zipfile

fname = 'http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'
target_dir = 'glove'

os.makedirs(target_dir, exist_ok=True)

names = [
    'glove.6B.100d.txt',
    'glove.6B.300d.txt',
]

with zipfile.ZipFile(cached_path(fname)) as z:
    for name in names:
        with z.open(name, 'r') as src:
            with open(os.path.join(target_dir, name), 'wb') as dst:
                shutil.copyfileobj(src, dst)
print(fname, '=>', names)
