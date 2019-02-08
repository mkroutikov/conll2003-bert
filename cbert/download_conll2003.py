from pytorch_pretrained_bert.file_utils import cached_path
import os
import shutil

files = [
    'https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testa',
    'https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testb',
    'https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.train',
]
target_dir = 'data/conll2003'

os.makedirs(target_dir, exist_ok=True)

for fname in files:
    name = os.path.basename(fname)
    shutil.copyfile(cached_path(fname), os.path.join(target_dir, name))
    print(fname, '=>', name)
