#!/usr/bin/env python

import tqdm

import torch
import torch.nn.functional as F
import torchtext
#from torchtext import data, datasets, vocab

from qtransformer import TextClassifier

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

    TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = torchtext.data.Field(sequential=False)
    train, test = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train, max_size=VOCAB_SIZE - 2)
    LABEL.build_vocab(train)

    train_iter, test_iter = torchtext.data.BucketIterator.splits((train, test), batch_size=BATCH_SIZE)

    print(f'Training examples: {len(train_iter)}')
    print(f'Testing examples:  {len(test_iter)}')

    model = TextClassifier(embed_dim=EMBED_DIM,
                           num_heads=NUM_HEADS,
                           num_blocks=NUM_TRANSFORMER_BLOCKS,
                           num_classes=NUM_CLS,
                           vocab_size=VOCAB_SIZE,
                           ff_dim=FF_DIM,
                           dropout=DROPOUT_RATE)
    opt = torch.optim.Adam(lr=LR, params=model.parameters())

    # training loop
    for iepoch in range(NUM_EPOCHS):
        print(f"Epoch {iepoch}/{NUM_EPOCHS+1}")
        model.train(True)

        for batch in tqdm.tqdm(train_iter):
            opt.zero_grad()
            input = batch.text[0]
            label = batch.label - 1

            if input.size(1) > MAX_SEQ_LEN:
                input = input[:, :MAX_SEQ_LEN]
            out = model(input)
            loss = F.nll_loss(out, label)

            loss.backward()
            opt.step()
        
        with torch.no_grad():
            model.train(False)
            tot, cor = 0.0, 0.0
            for batch in test_iter:
                input = batch.text[0]
                label = batch.label - 1
                if input.size(1) > MAX_SEQ_LEN:
                    input = input[:, :MAX_SEQ_LEN]
                out = model(input).argmax(dim=1)

                tot += float(input.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            print(f"Accuracy: {acc:.3}")
    print("End of training.")
    torch.save(model, "transformer_model.pt")
