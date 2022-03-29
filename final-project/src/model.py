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
        self.lstm_layer = torch.nn.LSTM(embedding_dim, lstm_hidden_dim)
        self.linear = torch.nn.Linear(
            in_features=lstm_hidden_dim, out_features=num_classes
        )

    def forward(self, seqs: torch.Tensor, seqs_lengths: torch.Tensor):
        embedding_output = self.embedding(seqs)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedding_output, seqs_lengths
        )
        lstm_output, (_, _) = self.lstm_layer(packed_embedded)
        ouput, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)

        max_pool = torch.nn.functional.adaptive_max_pool1d(
            ouput.permute(1, 2, 0), 1
        ).view(seqs.size(1), -1)

        linear_output = self.linear(torch.cat([max_pool], dim=1))

        return torch.nn.functional.log_softmax(linear_output, dim=1)
