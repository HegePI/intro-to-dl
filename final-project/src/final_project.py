import torch
import torchtext

import math

import news_dataset
import model

# Hyperparameters
N_EPOCHS = 15
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_DEV = 100
LR = 0.01


# --- fixed constants ---
NUM_CLASSES = 24
EMBEDDING_DIM = 200
LSTM_HIDDEN_DIM = 200
DATA_DIR = "final-project/data"
TOPICS = "final-project/topic_codes.txt"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = news_dataset.newsDataset(DATA_DIR, TOPICS)

    train = math.floor(len(data) * (4 / 5))
    test = math.floor((len(data) - train) / 2)
    dev = math.ceil((len(data) - train) / 2)

    train_data, test_data, dev_data = torch.utils.data.random_split(
        dataset=data, lengths=[train, test, dev]
    )

    txt_field = torchtext.data.Field(
        sequential=True, use_vocab=True, include_lengths=True
    )

    label_field = torchtext.data.Field(sequential=True, use_vocab=False)

    txt_field.build_vocab(
        train_data,
        dev_data,
        max_size=10_000,
        vectors="glove.twitter.27B.200d",
        unk_init=torch.Tensor.normal_,
    )

    label_field.build_vocab(train_data)

    train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(
        datasets=(train_data, dev_data, test_data),
        batch_sizes=(BATCH_SIZE_TRAIN, BATCH_SIZE_DEV, BATCH_SIZE_TEST),
        sort_key=lambda x: len(x),
        device=device,
        sort_within_batch=True,
        repeat=False,
    )

    PAD_IDX = txt_field.vocab.stoi[txt_field.pad_token]
    UNK_IDX = txt_field.vocab.stoi[txt_field.unk_token]

    lstm_model = model.Model(
        vocab_size=len(txt_field.vocab),
        embedding_dim=EMBEDDING_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    )

    pretrained_embeddings = txt_field.vocab.vectors
    lstm_model.embedding.weight.data.copy_(pretrained_embeddings)

    # Fix the <UNK> and <PAD> tokens in the embedding layer
    lstm_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    lstm_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)

    for batch_num, (data, target) in train_iter:
        print(data)
