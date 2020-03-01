import os
import pickle as pkl
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import pytrec_eval

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import init

from utils import data_processing, evaluate
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
        return 20000

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

        loss_pos = -F.logsigmoid(pos_similarities)
        loss_neg = -F.logsigmoid(-neg_similarities)

        return torch.mean(loss_pos + loss_neg)

    def inference_on_words(self, words):
        _words = []
        for w in words:
            if w not in self.vocab["token2id"]:
                continue
            _words.append(self.vocab["token2id"][w])
        _words = torch.tensor(_words, dtype=torch.long)
        if len(_words) == 0:
            print("Warning: Word embedding empty resulting in nans and meaningless results for that query!")

        return self.w_embeddings(_words)


class W2VRetrieval:

    def __init__(self, ARGS, w2v_model, docs):
        self.ARGS = ARGS
        self.model = w2v_model
        self.vocab = w2v_model.vocab
        self.docs = docs

    def match_query_against_words(self, query):
        query_repr = read_ap.process_text(query)
        q_embeddings = self.model.inference_on_words(query_repr)
        # If the query is a sentence we can compare the sentence against words
        agg_embeddings = aggregate_embeddings(q_embeddings, method=self.ARGS.aggr)

        _, sorted_w_idx = calc_cosine_similarity(agg_embeddings, self.model.w_embeddings.weight)

        results = [self.model.vocab["id2token"][i.item()] for i in sorted_w_idx[:self.ARGS.top_n]]
        return results

    def match_query_against_docs(self, query, doc_ids, doc_embeddings):
        query_repr = read_ap.process_text(query)
        q_embeddings = self.model.inference_on_words(query_repr)
        q_embedding = aggregate_embeddings(q_embeddings, method=self.ARGS.aggr)

        similarities, sorted_doc_idx = calc_cosine_similarity(q_embedding, doc_embeddings)

        results = [(doc_ids[i.item()], similarities[i.item()].item()) for i in sorted_doc_idx]
        return results

    def build_doc_embeddings(self):
        doc_ids, embed_docs = [], []
        for doc_id, doc in self.docs.items():
            embed_docs.append(self.get_doc_embedding(doc))
            doc_ids.append(doc_id)

        return doc_ids, torch.stack(embed_docs).squeeze()

    def get_doc_embedding(self, doc):
        w_embeddings = self.model.inference_on_words(doc)
        return aggregate_embeddings(w_embeddings, method=self.ARGS.aggr)

    def evaluate_queries(self, qrels, queries, save_path):

        save_path = os.path.join(self.ARGS.save_dir, save_path)

        overall_ser = {}
        # TODO: save necessary info for result file => trec_results = []

        doc_ids, doc_embeddings = self.build_doc_embeddings()

        print(f"Running Word2Vec Evaluation, ww-size: {self.ARGS.ww_size}, vocab-size: {len(self.vocab['id2token'])}")
        for qid in tqdm(qr  "map": 0.0003409593609120817,
  "ndcg": 0.2510072969802044
els):
            query_text = queries[qid]

            results = self.match_query_against_docs(query_text, doc_ids, doc_embeddings)
            overall_ser[qid] = dict(results)

            if int(qid) not in np.arange(76, 101):
                evaluate.write_trec_results(qid, results, save_path)

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
        metrics = evaluator.evaluate(overall_ser)

        evaluate.calc_mean_metrics(metrics)

        # dump this to JSON - *Not* Optional - This is submitted in the assignment!
        with open(os.path.join(save_path, "word2vec_metrics.json"), "w") as writer:
            json.dump(metrics, writer, indent=1)


def calc_cosine_similarity(query, goal_embeddings):
    similarities = F.cosine_similarity(query, goal_embeddings, dim=1)
    return similarities, similarities.argsort(descending=True)


def aggregate_embeddings(embeddings, method):
    if method == "mean":
        return embeddings.mean(dim=0)[None, ...]
    elif method == "min":
        return embeddings.min(dim=0).values[None, ...]
    elif method == "max":
        return embeddings.max(dim=0).values[None, ...]


def load_model(ARGS, model_path):
    print(f"Load model: {model_path}/model_final.pth...")
    model = torch.load(os.path.join(ARGS.save_dir, model_path, "model_final.pth"))
    return model.eval()


def train(ARGS, data_loader, model):
    optimizer = optim.SparseAdam(model.parameters(), lr=ARGS.lr)

    save_folder = os.path.join(ARGS.save_dir, "models", f"ww_{ARGS.ww_size}_{ARGS.freq_thresh}")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

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
            torch.save(model, os.path.join(save_folder, f"model_{t_iteration}.pth"))
    torch.save(model, os.path.join(save_folder, f"model_final.pth"))

    with open(os.path.join(save_folder, "losses.json"), 'w') as fp:
        json.dump(losses, fp)




# TODO:
#       T-tests on LSI models before after
#       4.3 run word2vec and doc2vec on correct test data
#       TQ 2.1 & 2.2