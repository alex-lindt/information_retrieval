import os
import pickle as pkl
import json
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import init

from utils import data_processing
import read_ap

from tf_idf import TfIdfRetrieval


class Word2VecDataset(Dataset):
    def __init__(self, data, ARGS):
        super().__init__()

        self.vocab = data["vocab"]
        self.token2id = self.vocab["token2id"]

        self.ww_size = ARGS.ww_size

        self.targets = data["target"]
        self.contexts = data["context"]
        self.negatives = data["negatives"]

        # Allows us to use data generated ww_size K with our current ww_size J given J < K
        self.clip = int((len(self.contexts[0]) - self.ww_size * 2) / 2)

        self.dataset_length = len(self.targets)
        print(f"Dataset Length: {self.dataset_length}")

    def __len__(self):
        # return self.dataset_length
        return 10000

    def __getitem__(self, idx):
        target = self.targets[idx]
        if self.clip > 0:
            context, negatives = self.contexts[idx][self.clip:-self.clip], self.negatives[idx][self.clip:-self.clip]
        else:
            context, negatives = self.contexts[idx], self.negatives[idx]

        _target = torch.tensor(self.token2id[target], dtype=torch.long)
        _context = torch.tensor([self.token2id[w] for w in context], dtype=torch.long)
        _negatives = torch.tensor([self.token2id[w] for w in negatives], dtype=torch.long)

        return _target, _context, _negatives


class Word2Vec(nn.Module):
    def __init__(self, vocab, embed_size):
        super(Word2Vec, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab["id2token"])
        self.embed_size = embed_size

        print("Vocabulary size", self.vocab_size)

        self.w_embeddings = nn.Embedding(self.vocab_size, embed_size, sparse=True)
        self.c_embeddings = nn.Embedding(self.vocab_size, embed_size, sparse=True)

        # Xavier init, best practice, otherwise learning is hindered.
        _r = 1.0 / self.embed_size
        init.uniform_(self.w_embeddings.weight.data, -_r, _r)
        init.uniform_(self.c_embeddings.weight.data, -_r, _r)

        # Does not work with embedding ...
        # torch.nn.init.xavier_uniform_(self.w_embeddings)
        # torch.nn.init.xavier_uniform_((self.C_embeddings)

    def forward(self, target, contexts, negatives):
        target_embeds = self.w_embeddings(target)
        context_embeds = self.c_embeddings(contexts)
        negative_embeds = self.c_embeddings(negatives)

        pos_similarities = torch.bmm(target_embeds[:, None, :], context_embeds.permute(0, 2, 1)).squeeze().sum(dim=1)
        neg_similarities = torch.bmm(target_embeds[:, None, :], negative_embeds.permute(0, 2, 1)).squeeze().sum(dim=1)

        # pos_similarities = torch.clamp(pos_similarities, max=10, min=-10)
        # neg_similarities = torch.clamp(neg_similarities, max=10, min=-10)

        loss_pos = -F.logsigmoid(pos_similarities)
        loss_neg = -F.logsigmoid(-neg_similarities)

        return torch.mean(loss_pos + loss_neg)

    def inference_on_words(self, words):
        _words = torch.tensor([self.vocab["token2id"][w] for w in words], dtype=torch.long)
        return self.w_embeddings(_words)


class W2VRetrieval:

    def __init__(self, ARGS, w2v_model, docs):
        index_path = f"./tfidf_index_{ARGS.freq_thresh}"
        if os.path.exists(index_path):
            index = data_processing.load_pickle(index_path)
            # inverted index
            self.ii = index["ii"]
        else:
            index = self.create_inverted_index(docs, index_path)
            self.ii = index["ii"]

        self.model = w2v_model
        self.vocab = w2v_model.vocab

    @staticmethod
    def create_inverted_index(docs, index_path):
        ii = defaultdict(list)
        df = defaultdict(int)

        doc_ids = list(docs.keys())

        print("Building Index")
        # build an inverted index
        for doc_id in tqdm(doc_ids):
            doc = docs[doc_id]

            counts = Counter(doc)
            for t, c in counts.items():
                ii[t].append((doc_id, c))
            # count df only once - use the keys
            for t in counts:
                df[t] += 1

        index = {
            "ii": ii,
            "df": df
        }
        data_processing.save_pickle(index, index_path)
        return index

    def match_query_against_words(self, query):
        query_repr = read_ap.process_text(query)
        q_embeddings = self.model.inference_on_words(query_repr)
        # If the query is a sentence we can compare the sentence against words
        agg_embeddings = aggregate_embeddings(q_embeddings, method="mean")

        similarities = F.cosine_similarity(agg_embeddings, self.model.w_embeddings.weight, dim=1)
        sort_indices = similarities.argsort(descending=True)

        for i in sort_indices[:10]:
            print(self.model.vocab["id2token"][i.item()])


def aggregate_embeddings(embeddings, method):
    if method == "mean":
        return embeddings.mean(dim=0)[None, ...]
    elif method == "min":
        return embeddings.min(dim=0).values[None, ...]
    elif method == "max":
        return embeddings.max(dim=0).values[None, ...]


def train(ARGS, data_loader, model):
    optimizer = optim.SparseAdam(model.parameters(), lr=ARGS.lr)

    if not os.path.exists(os.path.join(ARGS.save_dir, "models", f"ww_{ARGS.ww_size}")):
        os.makedirs(os.path.join(ARGS.save_dir, "models", f"ww_{ARGS.ww_size}"))

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

        if epoch in [int(ARGS.epochs * 0.25), int(ARGS.epochs * 0.5), int(ARGS.epochs * 0.75)]:
            torch.save(model, os.path.join(ARGS.save_dir, "models", f"ww_{ARGS.ww_size}", f"model_{t_iteration}.pth"))
    torch.save(model, os.path.join(ARGS.save_dir, "models", f"ww_{ARGS.ww_size}", f"model_final.pth"))

    with open(os.path.join(ARGS.save_dir, "models", f"ww_{ARGS.ww_size}", "losses.json"), 'w') as fp:
        json.dump(losses, fp)
