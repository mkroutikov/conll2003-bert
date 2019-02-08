import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel


class BidiLstmGloveModel(torch.nn.Module):

    def __init__(self,
        glove_embedding,
        labels_vocab_size,
        num_layers=2,
        rnn_size=100,
        embedding_dropout=0.5,
        output_dropout=0.5,
        lstm_dropout=0.5
    ):
        torch.nn.Module.__init__(self)

        embedding_dims = glove_embedding.shape[1]
        self.words_emb = torch.nn.Embedding(
            glove_embedding.shape[0],
            embedding_dims,
            padding_idx=0
        )
        self.words_emb.weight.data.copy_(torch.tensor(glove_embedding))

        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)

        self.rnn = torch.nn.LSTM(
            rnn_size,
            rnn_size,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.output_dropout = torch.nn.Dropout(output_dropout)

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

