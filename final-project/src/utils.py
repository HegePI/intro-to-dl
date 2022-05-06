# Auxilary functions for data preparation
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import spacy
import torch


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
