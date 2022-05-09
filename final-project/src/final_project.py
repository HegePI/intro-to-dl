import sys

import torch
import torchtext

from torchtext.legacy.data import Field
from torchtext.legacy.data import Pipeline

import model
import time
from parameters import Parameters

from utils import epoch_time, evaluate, get_optimizer, idx_to_multi_label_ohe, tokenizer


if __name__ == "__main__":

    params = Parameters()

    if len(sys.argv) > 1:
        params.set_mode(sys.argv[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    txt_field = Field(
        sequential=True, use_vocab=True, include_lengths=True, tokenize=tokenizer
    )

    label_field = Field(
        sequential=False,
        use_vocab=False,
        preprocessing=Pipeline(
            lambda x: idx_to_multi_label_ohe(x, params.get("num_classes"))
        ),
    )

    csv_fields = [("Labels", label_field), ("NewsText", txt_field)]

    train_data, dev_data, test_data = torchtext.legacy.data.TabularDataset.splits(
        path=params.get("data_path"),
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
        vectors=f"glove.6B.{params.get('embedding_dim')}d",
        unk_init=torch.Tensor.normal_,
    )

    label_field.build_vocab(train_data)

    train_iter, dev_iter, test_iter = torchtext.legacy.data.BucketIterator.splits(
        datasets=(train_data, dev_data, test_data),
        batch_sizes=(
            params.get("train_batch_size"),
            params.get("dev_batch_size"),
            params.get("test_batch_size"),
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
        embedding_dim=params.get("embedding_dim"),
        lstm_hidden_dim=params.get("lstm_hidden_dim"),
        num_classes=params.get("num_classes"),
    )

    pretrained_embeddings = txt_field.vocab.vectors
    lstm_model.embedding.weight.data.copy_(pretrained_embeddings)

    # Fix the <UNK> and <PAD> tokens in the embedding layer
    lstm_model.embedding.weight.data[UNK_IDX] = torch.zeros(params.get("embedding_dim"))
    lstm_model.embedding.weight.data[PAD_IDX] = torch.zeros(params.get("embedding_dim"))

    # BCELoss, when models last layer is sigmoid
    # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    # criterion = torch.nn.BCELoss()

    # BCEWithLogitsLoss, when models last layer is not sigmoid
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bceloss
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(128))
    # criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = get_optimizer(
        params.get("optimizer"),
        params=lstm_model.parameters(),
        lr=params.get("learning_rate"),
    )

    lstm_model = lstm_model.to(device)
    criterion = criterion.to(device)

    for epoch in range(params.get("n_epochs")):
        start_time = time.time()
        epoch_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0
        batch_number = 0

        lstm_model.train()

        for batch in train_iter:
            batch_number += 1
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
            print(batch_number)
            print("Precision:", 100 * train_precision / batch_number)
            print("Recall:", 100 * train_recall / batch_number)
            print("F1:", 100 * train_f1 / batch_number)
            print("Accuracy:", 100 * train_accuracy / batch_number)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        # print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        # print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    torch.save(lstm_model, params.get_mode())
