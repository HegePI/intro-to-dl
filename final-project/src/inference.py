import sys
import json
import re
import numpy as np
import spacy
import torch
import torchtext
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from torchtext.legacy.data import Field
from torchtext.legacy.data import Pipeline

tok = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"])


def tokenizer(s):
    return [w.text.lower() for w in tok(tweet_clean(s))]


def tweet_clean(text):
    """remove non alphanumeric character"""
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"https?:/\/\S+", " ", text)  # remove links
    text = re.sub(r"www?:/\/\S+", " ", text)
    return text.strip()


device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(
    "/home/heikki/koulu/intro-to-dl/final-project/src/adam_optimizer",
    map_location=torch.device("cpu"),
)
model.eval()

test_data = "/home/heikki/koulu/intro-to-dl/final-project/data/train.csv"

txt_field = Field(
    sequential=True, use_vocab=True, include_lengths=True, tokenize=tokenizer
)
csv_fields = [("NewsText", txt_field)]

test_data = torchtext.legacy.data.TabularDataset.splits(
    path="/home/heikki/koulu/intro-to-dl/final-project/data/",
    format="csv",
    train="train.csv",
    fields=csv_fields,
    skip_header=False,
)[0]

txt_field.build_vocab(
    test_data,
    max_size=100_000,
    vectors=f"glove.6B.200d",
    unk_init=torch.Tensor.normal_,
)

train_iter = torchtext.legacy.data.BucketIterator.splits(
    datasets=(test_data),
    batch_size=(10),
    sort_key=lambda x: len(x.NewsText),
    device=device,
    sort_within_batch=True,
    repeat=False,
)

PAD_IDX = txt_field.vocab.stoi[txt_field.pad_token]
UNK_IDX = txt_field.vocab.stoi[txt_field.unk_token]


pretrained_embeddings = txt_field.vocab.vectors
pretrained_embeddings = torch.nn.functional.pad(pretrained_embeddings, (0, 0, 0, 1403))
model.embedding.weight.data.copy_(pretrained_embeddings)

# Fix the <UNK> and <PAD> tokens in the embedding layer
model.embedding.weight.data[UNK_IDX] = torch.zeros(200)
model.embedding.weight.data[PAD_IDX] = torch.zeros(200)

model = model.to(device)


for batch in train_iter:

    seqs, seqs_lens = batch.dataset.NewsText

    out = model(seqs, seqs_lens)

    print(out)
