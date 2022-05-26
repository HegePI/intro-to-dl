import csv
import sys
import re
import spacy
import torch
import torchtext
import model_old

from parameters import Parameters

from torchtext.legacy.data import Field
from utils import get_codes_in_list, tokenizer, tweet_clean

print("starting inference.py")

device = "cuda" if torch.cuda.is_available() else "cpu"

params = Parameters()

if len(sys.argv) > 1:
    params.set_mode(sys.argv[1])

model = model_old.Model(
    vocab_size=100_002,
    embedding_dim=params.get("embedding_dim"),
    lstm_hidden_dim=params.get("lstm_hidden_dim"),
    num_classes=params.get("num_classes"),
)


print("loading model")
model.load_state_dict(torch.load(params.get_mode()))

model.eval()

print("model loaded")

id_field = Field(
    sequential=False,
    use_vocab=False,
)

txt_field = Field(
    sequential=True, use_vocab=True, include_lengths=True, tokenize=tokenizer
)

csv_fields = [("Labels", id_field), ("NewsText", txt_field)]

test_data = torchtext.legacy.data.TabularDataset.splits(
    path=params.get("data_path"),
    format="csv",
    train="data.csv",
    fields=csv_fields,
    skip_header=True,
)[0]

print("test data created")


txt_field.build_vocab(
    test_data,
    max_size=100_000,
    vectors=f'glove.6B.{params.get("embedding_dim")}d',
    unk_init=torch.Tensor.normal_,
)

id_field.build_vocab(test_data)

print("vocab created")

train_iter = torchtext.legacy.data.BucketIterator(
    dataset=test_data,
    batch_size=10,
    sort_key=lambda x: len(x.NewsText),
    device=device,
    sort_within_batch=True,
    repeat=False,
)

print("iterator created")

PAD_IDX = txt_field.vocab.stoi[txt_field.pad_token]
UNK_IDX = txt_field.vocab.stoi[txt_field.unk_token]


pretrained_embeddings = txt_field.vocab.vectors
pretrained_embeddings = torch.nn.functional.pad(pretrained_embeddings, (0, 0, 0, 1403))
model.embedding.weight.data.copy_(pretrained_embeddings)

# Fix the <UNK> and <PAD> tokens in the embedding layer
model.embedding.weight.data[UNK_IDX] = torch.zeros(params.get("embedding_dim"))
model.embedding.weight.data[PAD_IDX] = torch.zeros(params.get("embedding_dim"))

print("model finished")

model = model.to(device)

print("starting iterator")

with open(f'{params.get("data_path")}/results.csv', "a", newline="") as file:
    writer = csv.writer(file)
    line = ["NewsId"] + get_codes_in_list(params.get("topic_path"))
    writer.writerow(line)

    for batch in train_iter:

        seqs, seqs_lens = batch.NewsText
        ids = batch.Labels

        out = model(seqs, seqs_lens)

        out = (out > 0.5).int()

        for (id, sample) in zip(ids, out):
            print(id.item(), sample.tolist())
            writer.writerow([id.item()] + sample.tolist())
