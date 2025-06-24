
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from deepreaction.model.abstract_model import DeepReactionModel


# HierarchialAttentionNetwork
# TODO: Add docstring
class HAN(nn.Module, DeepReactionModel[Batch]):
    def __init__(
        self,
        embedding_in_channels: int,
        embedding_hidden_channels: int,
        gru_hidden_channels: int,
        class_num: int,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            embedding_in_channels, embedding_hidden_channels
        )
        self.word_gru = nn.GRU(
            input_size=embedding_hidden_channels,
            hidden_size=gru_hidden_channels,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.word_context = nn.Parameter(
            torch.empty(2 * gru_hidden_channels, 1)
        )
        nn.init.xavier_uniform_(self.word_context.data)
        self.word_dense = nn.Linear(
            2 * gru_hidden_channels, 2 * gru_hidden_channels
        )

        self.sentence_gru = nn.GRU(
            input_size=2 * gru_hidden_channels,
            hidden_size=gru_hidden_channels,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.sentence_context = nn.Parameter(
            torch.Tensor(2 * gru_hidden_channels, 1), requires_grad=True
        )
        nn.init.xavier_uniform_(self.sentence_context.data)
        self.sentence_dense = nn.Linear(
            2 * gru_hidden_channels, 2 * gru_hidden_channels
        )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(2 * gru_hidden_channels, class_num)

    def forward(self, x):
        sentence_num = x.shape[1]
        sentence_length = x.shape[2]
        x = x.view([-1, sentence_length])
        x = x.to(torch.int64)
        x_embedding = self.embedding(x)
        word_outputs, word_hidden = self.word_gru(x_embedding)
        word_outputs_attention = torch.tanh(self.word_dense(word_outputs))
        weights = torch.matmul(word_outputs_attention, self.word_context)
        weights = F.softmax(weights, dim=1)
        x = x.unsqueeze(2)
        weights = torch.where(
            x != 0,
            weights,
            torch.full_like(x, 0, dtype=torch.float),
        )
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)

        sentence_vector = torch.sum(word_outputs * weights, dim=1).view(
            [-1, sentence_num, word_outputs.shape[-1]]
        )
        sentence_outputs, sentence_hidden = self.sentence_gru(sentence_vector)
        attention_sentence_outputs = torch.tanh(
            self.sentence_dense(sentence_outputs)
        )
        weights = torch.matmul(
            attention_sentence_outputs, self.sentence_context
        )
        weights = F.softmax(weights, dim=1)
        x = x.view(-1, sentence_num, x.shape[1])
        x = torch.sum(x, dim=2).unsqueeze(2)
        weights = torch.where(
            x != 0,
            weights,
            torch.full_like(x, 0, dtype=torch.float),
        )
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)
        document_vector = torch.sum(sentence_outputs * weights, dim=1)
        document_vector = self.dropout(document_vector)
        output = self.fc(document_vector)
        return output
