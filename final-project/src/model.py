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

    def forward(self, seqs, seqs_lengths):
#removed typing eg. seqs: Torch.tensor
        embedding_output = self.embedding(seqs)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedding_output, seqs_lengths.cpu()
        )
    
#try to remove cpu, add lenghts[1]
        lstm_output, (hidden_state, _) = self.lstm_layer(packed_embedded)
        ouput, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)

        linear_output = self.linear(hidden_state[-1])
#out might need to have .cpu()
        out = torch.relu(linear_output)
        return out
