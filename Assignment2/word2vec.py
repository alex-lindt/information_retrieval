import os
import argparse
import pickle as pkl
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import init

from utils import data_processing


class Word2VecDataset(Dataset):
    def __init__(self, data):
        super().__init__()

        self.vocab = data["vocab"]
        self.w_to_idx = {w: i for i, w in enumerate(self.vocab)}

        self.targets = data["target"]
        self.contexts = data["context"]
        self.negatives = data["negatives"]

        self.dataset_length = len(self.targets)
        print(f"Dataset Length: {self.dataset_length}")

    def __len__(self):
        # return self.dataset_length
        return 10000

    def __getitem__(self, idx):
        target, context, negatives = self.targets[idx], self.contexts[idx], self.negatives[idx]

        _target = torch.tensor(self.w_to_idx[target], dtype=torch.long)
        _context = torch.tensor([self.w_to_idx[w] for w in context], dtype=torch.long)
        _negatives = torch.tensor([self.w_to_idx[w] for w in negatives], dtype=torch.long)

        return _target, _context, _negatives


class Word2Vec(nn.Module):
    def __init__(self, vocab, embed_size):
        super(Word2Vec, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_size = embed_size

        print("Vocabulary size", self.vocab_size)

        self.w_embeddings = nn.Embedding(self.vocab_size, embed_size, sparse=True)
        self.C_embeddings = nn.Embedding(self.vocab_size, embed_size, sparse=True)

        initrange = 1.0 / self.embed_size
        init.uniform_(self.w_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.C_embeddings.weight.data, 0)

    def forward(self, target, contexts, negatives):
        target_embeds = self.w_embeddings(target)
        context_embeds = self.C_embeddings(contexts)
        negative_embeds = self.C_embeddings(negatives)

        pos_similarities = torch.bmm(target_embeds[:, None, :], context_embeds.permute(0, 2, 1)).squeeze().sum(dim=1)
        neg_similarities = torch.bmm(target_embeds[:, None, :], negative_embeds.permute(0, 2, 1)).squeeze().sum(dim=1)

        pos_similarities = torch.clamp(pos_similarities, max=10, min=-10)
        neg_similarities = torch.clamp(neg_similarities, max=10, min=-10)

        loss_pos = -F.logsigmoid(pos_similarities)
        loss_neg = -F.logsigmoid(-neg_similarities)

        return torch.mean(loss_pos + loss_neg)


def train(ARGS, data_loader, model):
    optimizer = optim.SparseAdam(model.parameters(), lr=ARGS.lr)

    losses = {
        'epoch_losses': [],
        'total_losses': [],
        'mean_losses': []
    }
    t_iteration = 0
    for epoch in range(ARGS.epochs):
        losses['epoch_losses'] = []
        for iteration, (targets, contexts, negatives) in enumerate(data_loader):
            t_iteration += 1
            loss = model(targets, contexts, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses['epoch_losses'].append(loss.item())

        losses['total_losses'].append(losses['epoch_losses'])
        losses['mean_losses'].append(np.mean(losses['epoch_losses']))

        print(f"Epoch: {epoch}, Loss: {np.mean(losses['epoch_losses'])}, total iteration: {t_iteration}")

        if epoch in [ARGS.epochs * 0.25, ARGS.epochs * 0.5, ARGS.epochs * 0.75]:
            torch.save(model, os.path.join(ARGS.save_dir, "models", f"model_{t_iteration}.pth"))
    torch.save(model, os.path.join(ARGS.save_dir, "models", f"model_final.pth"))

    with open(os.path.join(ARGS.save_dir, "losses.json"), 'w') as fp:
        json.dump(losses, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--save-dir', type=str, default="./word2vec", help="Where outputs are saved")
    parser.add_argument('--filter-infreq-words', type=bool, default=False,
                        help="Run function filtering infrequent words")
    parser.add_argument('--use-data', type=int, default=20000, help="How much data will be used")

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--save-interval', type=int, default=1000000, help='save every save_interval iterations')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    # Word2vec
    parser.add_argument('--ww-size', type=int, default=4, help='Size of word window')
    parser.add_argument('--embed-dim', type=int, default=100, help='Size of word embedding')

    ARGS = parser.parse_args()

    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)
        os.makedirs(os.path.join(ARGS.save_dir, "models"))

    if ARGS.filter_infreq_words:
        data_processing.filter_infrequent_words(ARGS.use_data)

    data = data_processing.get_w2v_data(ARGS)

    vocab = data["vocab"]
    model = Word2Vec(vocab, ARGS.embed_dim).to(ARGS.device)

    word2vec_dataset = Word2VecDataset(data)
    data_loader = DataLoader(word2vec_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=2)

    train(ARGS, data_loader, model)
