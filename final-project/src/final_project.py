import sys
import json
import re
import spacy
import torch
import torchtext
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from torchtext.legacy.data import Field
from torchtext.legacy.data import Pipeline

import model
import time


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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluate(target, predicted):
    target = target.detach().cpu().numpy()
    prediction = torch.round(predicted).detach().cpu().numpy()

    batch_accuracy = accuracy_score(target, prediction)
    batch_precision = precision_score(target, prediction, average="micro")
    batch_recall = recall_score(target, prediction, average="micro")
    batch_f1_score = f1_score(target, prediction, average="micro")

    return batch_accuracy, batch_precision, batch_recall, batch_f1_score


def idx_to_multi_label_ohe(labels, n_classes):
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
    mode = "base"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    with open(
        "/home/markus/intro-to-dl/final-project/src/hyperparameters.json"
    ) as file:
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

    # BCELoss, when models last layer is sigmoid
    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    criterion = torch.nn.BCELoss()

    # BCEWithLogitsLoss, when models last layer is not sigmoid
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bceloss
    # criterion = torch.nn.BCEWithLogitsLoss()

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
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0

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

            batch_acc, batch_prec, batch_rec, batch_f1 = evaluate(targets, out)

            epoch_loss += float(loss)
            train_accuracy += batch_acc
            train_precision += batch_prec
            train_recall += batch_rec
            train_f1 += batch_f1

            print("Epoch_loss", epoch_loss)
            print("Accuracy:", 100 * train_accuracy)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        # print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        # print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    torch.save(lstm_model, mode)
