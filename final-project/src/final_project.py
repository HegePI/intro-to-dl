from typing import Dict
import re
import numpy as np
import spacy
import torch
import torchtext

from torchtext.legacy.data import Field
from torchtext.legacy.data import Pipeline

import model
import xml_to_csv

# Hyperparameters
N_EPOCHS = 15
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 10
BATCH_SIZE_DEV = 10
LR = 0.01


# --- fixed constants ---
NUM_CLASSES = 126
# Possible embedding dims: 50, 100, 200
EMBEDDING_DIM = 50
LSTM_HIDDEN_DIM = 200
DATA_DIR = "final-project/data"
TOPICS = "final-project/topic_codes.txt"


# Auxilary functions for data preparation
tok = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"])


def tokenizer(s):
    return [w.text.lower() for w in tok(tweet_clean(s))]


def tweet_clean(text):
    """remove non alphanumeric character"""
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"https?:/\/\S+", " ", text)  # remove links
    text = re.sub(r"www?:/\/\S+", " ", text)
    return text.strip()


def idx_to_multi_label_ohe(labels: list[int]) -> list[int]:
    labels = list(map(int, labels.split()))
    if len(labels) == 0:
        return torch.zeros(NUM_CLASSES).tolist()
    else:
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)
        labels = (
            torch.zeros(labels.size(0), NUM_CLASSES)
            .scatter_(1, labels, 1)
            .squeeze(0)
            .tolist()
        )
        return labels


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_processor = xml_to_csv.XmlToCsvWriter(DATA_DIR, TOPICS)

    # codes = data_processor.get_codes()

    # train_csv, dev_csv, test_csv = data_processor.write_data_to_csv_files(
    #     csv_sizes=[7 / 10, 2 / 10, 1 / 10]
    # )

    txt_field = Field(
        sequential=True, use_vocab=True, include_lengths=True, tokenize=tokenizer
    )

    label_field = Field(
        sequential=False,
        use_vocab=False,
        preprocessing=Pipeline(lambda x: idx_to_multi_label_ohe(x)),
    )

    csv_fields = [("Labels", label_field), ("NewsText", txt_field)]

    train_data, dev_data, test_data = torchtext.legacy.data.TabularDataset.splits(
        path="/home/heikki/koulu/intro-to-dl/final-project/data",
        format="csv",
        train="train.csv",
        validation="dev.csv",
        test="test.csv",
        fields=csv_fields,
        skip_header=False,
    )

    txt_field.build_vocab(
        train_data,
        dev_data,
        max_size=100_000,
        vectors=f"glove.6B.{EMBEDDING_DIM}d",
        unk_init=torch.Tensor.normal_,
    )

    label_field.build_vocab(train_data)

    train_iter, dev_iter, test_iter = torchtext.legacy.data.BucketIterator.splits(
        datasets=(train_data, dev_data, test_data),
        batch_sizes=(BATCH_SIZE_TRAIN, BATCH_SIZE_DEV, BATCH_SIZE_TEST),
        sort_key=lambda x: len(x.NewsText),
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

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)

    lstm_model = lstm_model.to(device)
    criterion = criterion.to(device)

    for epoch in range(N_EPOCHS):

        lstm_model.train()

        for batch in train_iter:
            optimizer.zero_grad()

            seqs, seqs_lens = batch.NewsText
            targets = batch.Labels.float()

            print(targets.shape)

            out = lstm_model(seqs, seqs_lens)

            print(out.shape)

            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
