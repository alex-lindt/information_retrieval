import os

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
    def __init__(self, data, ARGS):
        super().__init__()

        self.vocab = data["vocab"]
        self.w_to_idx = {w: i for i, w in enumerate(self.vocab)}

        self.ww_size = ARGS.ww_size

        self.targets = data["target"]
        self.contexts = data["context"]
        self.negatives = data["negatives"]

        self.clip_size = len(self.contexts[0]) - self.ww_size

        print(len(self.contexts[0]), self.ww_size, self.clip_size)

        self.dataset_length = len(self.targets)
        print(f"Dataset Length: {self.dataset_length}")

    def __len__(self):
        # return self.dataset_length
        return 10000

    def __getitem__(self, idx):
        target, context, negatives = self.targets[idx], self.contexts[idx], self.negatives[idx]

        _target = torch.tensor(self.vocab.token2id[target], dtype=torch.long)
        _context = torch.tensor([self.vocab.token2id[w] for w in context], dtype=torch.long)
        _negatives = torch.tensor([self.vocab.token2id[w] for w in negatives], dtype=torch.long)

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


        # Xavier init, best practice over various repositories. Otherwise learning is hindered.
        # Replace with torch.nn.init.xavier_normal_ ?
        _d = 1.0 / self.embed_size
        init.uniform_(self.w_embeddings.weight.data, -_d, _d)
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
