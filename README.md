# conll2003-bert
Applying BERT neural net to CoNLL2003 NER task

## Setting things up

Create Python3 virtual envirnoment and install dependencies
```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Download CONLL2003 dataset
```
python -m cbert.download_conll2003
```

## Download Glove (optional)

Skip this if you not interested in trying models with Glove.

```
python -m cbert.download_glove
```

## Run training

1. Glove + simple one-layer bidi LSTM
   ```
   python -m cbert.train traindir-A
   ```
   This will create directory `traindir`. Checkpoints, tensorboard stats, etc will be saved there

2. BERT as an embedding (frozen) + learning a simple one-layer bidi LSTM on top
   ```
   python -m cbert.train01 traindir-B
   ```

3. BERT with a simple one-layer bidi LSTM on top (all layers are learned)
   ```
   python -m cbert.train02 traindir-B
   ```

4. Glove embedings + ADW-LSTM from fastai (not working, lacking bidi support at the moment)
   ```
   python -m cbert.train03 traindir-C
   ```

5. BERT tagger (nothing on top - just a dense layer). All layers are trained.
   ```
   python -m cbert.train04 traindir-D
   ```

## Tensorboard visualization

Training, validation, and test statistics is written to the train dir and can be viewed with `tensorboard`.

```
pip install tensorboard
tensorboard --logdir .
```

## Experiments

1. Baseline bidi LSTM + Glove: dev_F1=92.7, test_F1=88.9
2. BERT as embedding: 
3. BERT + bidi LSTM: dev_F1: 94.5, test_F1: 90.0
4. AWD LSTM + Glove:
5. BERT tagger: dev_F1: 95.6, test_F1: 91.0
