#!/usr/bin/env python

import argparse
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

        inputs = torch.LongTensor(batch.text[0])
        if inputs.size(1) > MAX_SEQ_LEN:
            inputs = inputs[:, :MAX_SEQ_LEN]
        predictions = model(inputs).squeeze(1)
        
        label = batch.label - 1
        #label = label.unsqueeze(1)
        loss = criterion(predictions, label)
        #loss = F.nll_loss(predictions, label)
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
            inputs = torch.LongTensor(batch.text[0])
            if inputs.size(1) > MAX_SEQ_LEN:
                inputs = inputs[:, :MAX_SEQ_LEN]
            predictions = model(inputs).squeeze(1)
            
            label = batch.label - 1
            #label = label.unsqueeze(1)
            loss = criterion(predictions, label)
            #loss = F.nll_loss(predictions, label)
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-B', '--batch_size', default=32, type=int)
    parser.add_argument('-E', '--n_epochs', default=5, type=int)
    parser.add_argument('-C', '--n_classes', default=2, type=int)
    parser.add_argument('-l', '--lr', default=0.001, type=float)
    parser.add_argument('-v', '--vocab_size', default=20000, type=int)
    parser.add_argument('-e', '--embed_dim', default=8, type=int)
    parser.add_argument('-s', '--max_seq_len', default=64, type=int)
    parser.add_argument('-f', '--ffn_dim', default=8, type=int)
    parser.add_argument('-t', '--n_transformer_blocks', default=1, type=int)
    parser.add_argument('-H', '--n_heads', default=2, type=int)
    parser.add_argument('-q', '--n_qubits', default=0, type=int)
    parser.add_argument('-L', '--n_qlayers', default=1, type=int)
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float)
    args = parser.parse_args()

    MAX_SEQ_LEN = args.max_seq_len

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    #LABEL = data.Field(sequential=False)
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    print(f'Training examples: {len(train_data)}')
    print(f'Testing examples:  {len(test_data)}')

    TEXT.build_vocab(train_data, max_size=args.vocab_size - 2)  # exclude <UNK> and <PAD>
    LABEL.build_vocab(train_data)

    train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_size=args.batch_size)
    
    model = TextClassifier(embed_dim=args.embed_dim,
                           num_heads=args.n_heads,
                           num_blocks=args.n_transformer_blocks,
                           num_classes=args.n_classes,
                           vocab_size=args.vocab_size,
                           ffn_dim=args.ffn_dim,
                           n_qubits=args.n_qubits,
                           n_qlayers=args.n_qlayers,
                           dropout=args.dropout_rate)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    if args.n_classes < 3:
        criterion = torch.nn.BCEWithLogitsLoss()  # logits -> sigmoid -> loss
    else:
        criterion = torch.nn.CrossEntropyLoss()  # logits -> log_softmax -> NLLloss

    # training loop
    best_valid_loss = float('inf')
    for iepoch in range(args.n_epochs):
        start_time = time.time()

        print(f"Epoch {iepoch}/{args.n_epochs+1}")

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
