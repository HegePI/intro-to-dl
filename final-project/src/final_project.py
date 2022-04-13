import sys
import json
import re
import numpy as np
import spacy
import torch
import torchtext

from torchtext.legacy.data import Field
from torchtext.legacy.data import Pipeline

import model
import time


# Auxilary functions for data preparation
tok = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"])


def tokenizer(s: str) -> str:
    return [w.text.lower() for w in tok(tweet_clean(s))]


def tweet_clean(text: str) -> str:
    """remove non alphanumeric character"""
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"https?:/\/\S+", " ", text)  # remove links
    text = re.sub(r"www?:/\/\S+", " ", text)
    return text.strip()


def epoch_time(start_time: int, end_time: int) -> int:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_accuracy(output: torch.Tensor, gold: torch.Tensor) -> float:
    _, predicted = torch.max(output, dim=1)
    correct = torch.sum(torch.eq(predicted, gold)).item()
    acc = correct / gold.shape[0]
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.NewsText
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.Labels)
            acc = get_accuracy(predictions, batch.Labels)
            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def idx_to_multi_label_ohe(labels: list[int], n_classes: int) -> list[int]:
    labels = list(map(int, labels.split()))
    if len(labels) == 0:
        return torch.zeros(n_classes).tolist()
    else:
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)
        labels = (
            torch.zeros(labels.size(0), n_classes)
            .scatter_(1, labels, 1)
            .squeeze(0)
            .tolist()
        )
        return labels


def get_optimizer(optimizer, params, lr):
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr)
    elif optimizer == "adam":
        return torch.optim.Adam(params, lr)
    elif optimizer == "rms_prop":
        return torch.optim.RMSprop(params, lr)
    else:
        print("No optimizer specified in hyperparameters, defaulting to SGD")
        return torch.optim.SGD(params, lr)


if __name__ == "__main__":
    if len(sys.argv) == 0:
        print("no mode defined, terminating")
        sys.exit()

    mode = sys.argv[1]

    with open("final-project/src/hyperparameters.json") as file:
        params = json.loads(file.read())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    txt_field = Field(
        sequential=True, use_vocab=True, include_lengths=True, tokenize=tokenizer
    )

    label_field = Field(
        sequential=False,
        use_vocab=False,
        preprocessing=Pipeline(
            lambda x: idx_to_multi_label_ohe(x, params[mode]["num_classes"])
        ),
    )

    csv_fields = [("Labels", label_field), ("NewsText", txt_field)]

    train_data, dev_data, test_data = torchtext.legacy.data.TabularDataset.splits(
        path=params[mode]["data_path"],
        format="csv",
        train="train.csv",
        validation="dev.csv",
        test="test.csv",
        fields=csv_fields,
        skip_header=False,
    )

    ed = params[mode]["embedding_dim"]
    txt_field.build_vocab(
        train_data,
        dev_data,
        max_size=100_000,
        vectors=f"glove.6B.{ed}d",
        unk_init=torch.Tensor.normal_,
    )

    label_field.build_vocab(train_data)

    train_iter, dev_iter, test_iter = torchtext.legacy.data.BucketIterator.splits(
        datasets=(train_data, dev_data, test_data),
        batch_sizes=(
            params[mode]["train_batch_size"],
            params[mode]["dev_batch_size"],
            params[mode]["test_batch_size"],
        ),
        sort_key=lambda x: len(x.NewsText),
        device=device,
        sort_within_batch=True,
        repeat=False,
    )

    PAD_IDX = txt_field.vocab.stoi[txt_field.pad_token]
    UNK_IDX = txt_field.vocab.stoi[txt_field.unk_token]

    lstm_model = model.Model(
        vocab_size=len(txt_field.vocab),
        embedding_dim=params[mode]["embedding_dim"],
        lstm_hidden_dim=params[mode]["lstm_hidden_dim"],
        num_classes=params[mode]["num_classes"],
    )

    pretrained_embeddings = txt_field.vocab.vectors
    lstm_model.embedding.weight.data.copy_(pretrained_embeddings)

    # Fix the <UNK> and <PAD> tokens in the embedding layer
    lstm_model.embedding.weight.data[UNK_IDX] = torch.zeros(
        params[mode]["embedding_dim"]
    )
    lstm_model.embedding.weight.data[PAD_IDX] = torch.zeros(
        params[mode]["embedding_dim"]
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = get_optimizer(
        params[mode]["optimizer"],
        params=lstm_model.parameters(),
        lr=params[mode]["learning_rate"],
    )

    lstm_model = lstm_model.to(device)
    criterion = criterion.to(device)

    for epoch in range(params[mode]["n_epochs"]):
        start_time = time.time()
        epoch_loss = 0
        epoch_acc = 0

        correct = 0
        total = 0

        lstm_model.train()

        for batch in train_iter:
            optimizer.zero_grad()

            seqs, seqs_lens = batch.NewsText
            targets = batch.Labels.float()

            out = lstm_model(seqs, seqs_lens)

            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            # detach the tensors, numpy doesn't play well when a gradient is attached to a tensor
            predictions = out.detach().numpy()
            predicted = np.round(predictions)  # round 0.49 and smaller to 0, 0.5 to 1.
            total += targets.size(
                1
            )  # Total number of predictions. Not sure if this is the right amount ??

            correct += (predicted == targets.numpy().astype(int)).sum()

        # Not functional yet
        train_loss, train_acc = (
            epoch_loss / len(train_iter),
            epoch_acc / len(train_iter),
        )
        # valid_loss, valid_acc = evaluate(model, dev_iter, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        accuracy = correct / total
        print("Accuracy: {}%".format(accuracy))

        print(f"Train Acc: {train_acc*100:.2f}%")

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        # print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        # print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
