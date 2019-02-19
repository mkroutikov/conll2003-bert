import logging
from types import SimpleNamespace as ns
import torch.nn.functional as F
import torch
from .models import AwdLstmGlove
from .batcher import Batcher
from .metrics import TokenAndRecordAccuracy, F1Score, CrossEntropyLoss, MetricSet
from .feats import Feats
from .util import read_conll2003, save_json, SummaryWriter
from tqdm import tqdm, trange


logging.basicConfig(level=logging.INFO)

def train(
    traindir,
    *,
    datadir = 'data/conll2003',
    epochs  = 100,
):
    save_json(locals(), f'{traindir}/params.json')

    summary_writer = SummaryWriter(f'{traindir}/summaries')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = list(read_conll2003(f'{datadir}/eng.train'))
    testa_data = list(read_conll2003(f'{datadir}/eng.testa'))
    testb_data = list(read_conll2003(f'{datadir}/eng.testb'))

    feats = Feats()
    train, testa, testb = feats.encode(train_data, testa_data, testb_data)
    feats.save(traindir)  # save vocabularies

    words_vocab = feats.vocab['words']
    labels_vocab = feats.vocab['labels']

    train_batches = Batcher(train, batch_size=32, shuffle=True).to(DEVICE)
    testa_batches = Batcher(testa, batch_size=32).to(DEVICE)
    testb_batches = Batcher(testb, batch_size=32).to(DEVICE)

    model = AwdLstmGlove(
        glove_filename='glove/glove.6B.100d.txt',
        words_vocab=words_vocab,
        labels_vocab_size=len(labels_vocab),
        num_layers=1
    ).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=1.5, momentum=0.0, weight_decay=0.000001)

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1. / (1. + 0.05 * epoch))

    eval_metric = MetricSet({
        'acc': TokenAndRecordAccuracy(),
        'entity': F1Score(labels_vocab=labels_vocab),
        'loss': CrossEntropyLoss(),
    })
    train_metric = MetricSet({
        'acc': TokenAndRecordAccuracy()
    })

    def cross_entropy(x, labels):
        x = x.transpose(1, 2)
        return F.cross_entropy(x, labels, ignore_index=0, reduction='mean')

    global_step = 0
    best = ns(f1=0.0, epoch=-1)
    for epoch in trange(epochs, desc='epoch'):
        schedule.step()

        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        train_metric.reset()
        for count, (words, labels) in enumerate(tqdm(train_batches, desc='step'), start=1):
            optimizer.zero_grad()

            x = model(words)
            loss = cross_entropy(x, labels)
            loss.backward()
            optimizer.step()

            # accumulate metrics
            train_metric.append(x, labels, loss=loss.item())

            global_step += 1
            if count % 50 == 0:
                summ = train_metric.summary
                tqdm.write(
                    f'epoch: {epoch:5}, step: {count:6}, loss: {summ["acc.loss"]:10.4f} '
                    f'tacc: {summ["acc.tacc"]:6.4f}, racc: {summ["acc.racc"]:6.4f}'
                )
                summary_writer['train'].add_scalar_metric(summ, global_step=global_step)

    with torch.no_grad():
        model.train(False)
        eval_metric.reset().update((model(x), y) for x,y in tqdm(testa_batches, desc='dev'))
        summ = eval_metric.summary

        f1 = summ['entity.f1']
        if f1 > best.f1:
            best.f1 = f1
            best.epoch = epoch
            torch.save(model, f'{traindir}/model.pickle')

        tqdm.write(
            f'Dev: loss: {summ["loss.loss"]:6.4f}, tacc: {summ["acc.tacc"]:6.4f}, racc: {summ["acc.racc"]:6.4f}, '
            f'entity.f1: {summ["entity.f1"]:6.4f}, best.f1: {best.f1:6.4f} at epoch {best.epoch}'
        )
        summary_writer['dev'].add_scalar_metric(summ, global_step=global_step)
        model.train(True)

    model = torch.load(f'{traindir}/model.pickle')

    with torch.no_grad():
        model.train(False)

        metric = MetricSet({
            'acc'    : TokenAndRecordAccuracy(),
            'entity' : F1Score(labels_vocab=labels_vocab),
            'viterbi': F1Score(labels_vocab=labels_vocab, entity_decoder='viterbi'),  # this is sloow
            'loss'   : CrossEntropyLoss(),
        })

        tqdm.write('Evaluating the best model on testa')
        metric.reset().update((model(x), y) for x,y in tqdm(testa_batches, desc="dev"))
        tqdm.write(repr(metric.summary))
        summary_writer['final-dev'].add_scalar_metric(metric.summary)

        tqdm.write('Evaluating the best model on testb')
        metric.reset().update((model(x), y) for x,y in tqdm(testb_batches, desc="test"))
        tqdm.write(repr(metric.summary))
        summary_writer['final-train'].add_scalar_metric(metric.summary)


if __name__ == '__main__':
    import fire
    fire.Fire(train)
