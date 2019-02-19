import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from .glove import Glove
import logging
import math



class BidiLstmGloveModel(torch.nn.Module):

    def __init__(self,
        glove_filename,
        words_vocab,
        labels_vocab_size,
        num_layers=2,
        rnn_size=100,
        dropout=0.5
    ):
        torch.nn.Module.__init__(self)

        glove = Glove.load(glove_filename)
        logging.info('Loaded %s Glove vectors', len(glove))

        glove_embedding = np.array([get_embedding(glove, x) for x in words_vocab.values])
        embedding_dims = glove_embedding.shape[1]
        self.words_emb = torch.nn.Embedding(
            glove_embedding.shape[0],
            embedding_dims,
            padding_idx=0
        )
        self.words_emb.weight.data.copy_(torch.tensor(glove_embedding))

        self.embedding_dropout = torch.nn.Dropout(dropout)

        self.rnn = torch.nn.LSTM(
            embedding_dims,
            rnn_size,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.output_dropout = torch.nn.Dropout(dropout)

        self.output = torch.nn.Linear(rnn_size*2, labels_vocab_size)  # *2 because of bidi

    def forward(self, words):
        '''
        words - (B, S) containing word ids. 0 - padding, 1 - OOV
        chars - (B, S, W) containing chracter ids. 0 - padding, 1 - OOV
        '''

        mask = (words > 0).float().unsqueeze(dim=2)
        sentence_lengths = (words > 0).long().sum(dim=1)
        permutation = torch.argsort(sentence_lengths, descending=True)

        x = self.words_emb(words)
        x = self.embedding_dropout(x)

        x = x[permutation]
        sentence_lengths = sentence_lengths[permutation]

        x = pack_padded_sequence(x, lengths=sentence_lengths, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        reverse_permutation = torch.argsort(permutation)
        x = x[reverse_permutation]

        x = self.output_dropout(x)

        x = self.output(x)

        return x


def get_embedding(glove, word):
    if word in glove:
        return glove[word]
    elif word.lower() in glove:
        return glove[word.lower()]
    else:
        return get_random_embedding(glove.dim)

def get_random_embedding(dim):
    scale = math.sqrt(3./dim)
    return np.random.uniform(-scale, scale, size=(dim,))


class BidiLstmBertModel(torch.nn.Module):

    def __init__(self,
        bert_model,
        rnn_size,
        labels_vocab_size,
        num_layers=2,
        dropout=0.5
    ):
        torch.nn.Module.__init__(self)

        self.bert = BertModel.from_pretrained(bert_model)

        self.dropout = torch.nn.Dropout(dropout)

        bert_size = self.bert.config.hidden_size

        self.rnn = torch.nn.LSTM(
            bert_size,
            rnn_size,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.output = torch.nn.Linear(rnn_size*2, labels_vocab_size)  # *2 because of bidi

    def forward(self, words):
        '''
        words - (B, S) containing word ids. 0 - padding, 1 - OOV
        chars - (B, S, W) containing chracter ids. 0 - padding, 1 - OOV
        '''

        mask = (words > 0).float()
        sentence_lengths = (words > 0).long().sum(dim=1)
        permutation = torch.argsort(sentence_lengths, descending=True)

        x,_ = self.bert(words, attention_mask=mask, output_all_encoded_layers=False)
        x = self.dropout(x)

        x = x[permutation]
        sentence_lengths = sentence_lengths[permutation]

        x = pack_padded_sequence(x, lengths=sentence_lengths, batch_first=True)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        reverse_permutation = torch.argsort(permutation)
        x = x[reverse_permutation]

        x = self.dropout(x)

        x = self.output(x)

        return x


class BertTagger(torch.nn.Module):

    def __init__(self,
        bert_model,
        labels_vocab_size,
        dropout=0.5
    ):
        torch.nn.Module.__init__(self)

        self.bert = BertModel.from_pretrained(bert_model)

        self.dropout = torch.nn.Dropout(dropout)

        bert_size = self.bert.config.hidden_size
        self.output = torch.nn.Linear(bert_size, labels_vocab_size) 

    def forward(self, words):
        '''
        words - (B, S) containing word ids. 0 - padding, 1 - OOV
        chars - (B, S, W) containing chracter ids. 0 - padding, 1 - OOV
        '''
        mask = (words > 0).float()
        x,_ = self.bert(words, attention_mask=mask, output_all_encoded_layers=False)
        x = self.dropout(x)
        x = self.output(x)

        return x


from fastai.text.models.awd_lstm import AWD_LSTM

class AwdLstmGlove(torch.nn.Module):

    def __init__(self,
        glove_filename,
        words_vocab,
        labels_vocab_size,
        num_layers=2,
        rnn_size=100,
        hidden_dropout=0.2,
        input_dropout=0.6,
        embed_dropout=0.1,
        weight_dropout=0.1,
        qrnn=False,  # this requires CUDA
        output_dropout=0.5,
    ):
        torch.nn.Module.__init__(self)

        glove = Glove.load(glove_filename)
        logging.info('Loaded %s Glove vectors', len(glove))

        glove_embedding = np.array([get_embedding(glove, x) for x in words_vocab.values])
        embedding_dims = glove_embedding.shape[1]

        self.awd = AWD_LSTM(
            glove_embedding.shape[0],
            embedding_dims,
            rnn_size,
            num_layers,
            pad_token = 0,
            hidden_p=hidden_dropout,
            input_p=input_dropout,
            embed_p=embed_dropout,
            weight_p=weight_dropout,
            qrnn=qrnn
        )
        self.awd.encoder.weight.data.copy_(torch.tensor(glove_embedding))

        self.output_dropout = torch.nn.Dropout(output_dropout)

        self.output = torch.nn.Linear(embedding_dims, labels_vocab_size)  # *2 because of bidi

    def forward(self, words):
        '''
        words - (B, S) containing word ids. 0 - padding, 1 - OOV
        chars - (B, S, W) containing chracter ids. 0 - padding, 1 - OOV
        '''

        _, x = self.awd(words)
        x = x[-1]  # pick top layer

        x = self.output_dropout(x)

        x = self.output(x)

        return x

