import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_hidden_dim: int,
        num_classes: int,
    ):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm_layer = torch.nn.LSTM(
            input_size=embedding_dim, hidden_size=lstm_hidden_dim
        )
        self.linear1 = torch.nn.Linear(in_features=lstm_hidden_dim, out_features=512)
        self.linear2 = torch.nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, seqs, seqs_lengths):

        embedding_output = self.embedding(seqs)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedding_output, seqs_lengths.cpu()
        )

        _, (hidden_state, _) = self.lstm_layer(packed_embedded)
        # ouput, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)

        out = self.linear1(hidden_state[-1])
        out = torch.sigmoid(out)
        out = self.linear2(out)
        # out = torch.sigmoid(out)
        return out
