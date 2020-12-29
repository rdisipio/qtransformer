#!/usr/bin/env python

import tqdm
import time

import torch
import torch.nn.functional as F
import torchtext
from torchtext import data, datasets, vocab

from qtransformer import TextClassifier


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        inputs = batch.text[0]
        if inputs.size(1) > MAX_SEQ_LEN:
            inputs = inputs[:, :MAX_SEQ_LEN]
        predictions = model(inputs).squeeze(1)
        
        label = batch.label - 1
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            inputs = batch.text[0]
            if inputs.size(1) > MAX_SEQ_LEN:
                inputs = inputs[:, :MAX_SEQ_LEN]
            predictions = model(inputs).squeeze(1)
            
            label = batch.label - 1
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    VOCAB_SIZE = 20000
    MAX_SEQ_LEN = 128
    EMBED_DIM = 32
    NUM_HEADS = 4
    NUM_TRANSFORMER_BLOCKS = 1
    NUM_CLS = 2
    FF_DIM = 16
    DROPOUT_RATE = 0.1
    LR = 0.001

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    #LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train_data, max_size=VOCAB_SIZE - 2)  # exclude <UNK> and <PAD>
    LABEL.build_vocab(train_data)

    train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_size=BATCH_SIZE)
    print(f'Training examples: {len(train_iter)}')
    print(f'Testing examples:  {len(test_iter)}')

    model = TextClassifier(embed_dim=EMBED_DIM,
                           num_heads=NUM_HEADS,
                           num_blocks=NUM_TRANSFORMER_BLOCKS,
                           num_classes=NUM_CLS,
                           vocab_size=VOCAB_SIZE,
                           ff_dim=FF_DIM,
                           dropout=DROPOUT_RATE)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(lr=LR, params=model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()  # logits -> sigmoid -> loss

    # training loop
    best_valid_loss = float('inf')
    for iepoch in range(NUM_EPOCHS):
        start_time = time.time()

        print(f"Epoch {iepoch}/{NUM_EPOCHS+1}")

        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
        
        print(f'Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
