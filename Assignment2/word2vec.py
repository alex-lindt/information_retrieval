import os
import argparse
import json
import random
import pickle as pkl
from collections import defaultdict, Counter

import numpy as np
import pytrec_eval
from tqdm import tqdm

import torch
import torch.nn as nn

import read_ap
import download_ap

from nltk.corpus import stopwords


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, target, context):
        pass


def get_w2v_data(ARGS):
    data_path = os.path.join(ARGS.save_dir, "data")
    if os.path.exists(data_path):
        with open(data_path, "rb") as reader:
            data = pkl.load(reader)
            if data["ww_size"] != ARGS.ww_size:
                print(f"Error! word window size of data ({data['ww_size']}) does not match the input ({ARGS.ww_size})")
                # Could terminate here ...
    else:
        data = build_w2v_data(data_path, ARGS)

    return data


def build_w2v_data(data_path, ARGS):
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs("processed_docs_filtered")

    print(len(docs_by_id))

    total_vocab = set()
    # Head start total_vocab for negative sampling
    for i, (doc_id, doc) in enumerate(docs_by_id.items()):
        if i > 15:
            break
        doc_set = set(doc)
        total_vocab.update(doc_set)

    ww_size = ARGS.ww_size

    data = {
        "target": [],
        "context": [],
        "negatives": [],
        "ww_size": ww_size,
        "vocab": set()
    }

    # Create instance for retrieval
    for i, (doc_id, doc) in tqdm(enumerate(docs_by_id.items())):

        doc_length = len(doc)

        # update set for total vocab
        doc_set = set(doc)
        total_vocab.update(doc_set)

        # we don't consider the first ww_size words as it doesn't make a difference in the limit
        for i_target in range(ww_size, doc_length - ww_size):
            word_context = []
            word_context.extend(doc[i_target - ww_size:i_target])
            word_context.extend(doc[i_target+1:i_target + ww_size+1])

            target = doc[i_target]

            # negative sampling
            k = ww_size * 2
            negative_sample = random.sample(total_vocab, k)

            data["target"].append(target)
            data["context"].append(word_context)
            data["negatives"].append(negative_sample)

    data["vocab"] = total_vocab

    with open(data_path, "wb") as writer:
        pkl.dump(data, writer)

    return data


def filter_infrequent_words():
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    infreq_list = []
    general_counts = Counter()

    # filter frequent words
    for i, (doc_id, doc) in tqdm(enumerate(docs_by_id.items())):
        if i > 20000:
            break

        counts = Counter(doc)
        general_counts += counts

    for t, c in tqdm(general_counts.items()):
        if c <= 50:
            infreq_list.append(t)

    filtered_docs_by_id = {}
    for i, (doc_id, doc) in tqdm(enumerate(docs_by_id.items())):
        if i > 20000:
            break
        _removed = [word for word in doc if word not in infreq_list]
        filtered_docs_by_id[doc_id] = _removed

    with open(f"./processed_docs_filtered.pkl", "wb") as writer:
        pkl.dump(filtered_docs_by_id, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--save_dir', type=str, default="./word2vec", help="Where outputs are saved")
    parser.add_argument('--filter_infreq_words', type=bool, default=False,
                        help="Run function filtering infrequent words")
    parser.add_argument('--use_data', type=int, default=80000, help="How much data will be used")

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--save_interval', type=int, default=500, help='save every save_interval iterations')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    # Word2vec
    parser.add_argument('--ww_size', type=int, default=4, help='Size of word window')

    ARGS = parser.parse_args()

    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    if ARGS.filter_infreq_words:
        filter_infrequent_words()

    data = get_w2v_data(ARGS)

# loss_function = nn.NLLLoss()

# vocab = set(raw_text)
# vocab_size = len(vocab)
# word_to_ix = {word: i for i, word in enumerate(vocab)}

# data = []
# for i in range(2, len(raw_text) - 2):
#     context = [raw_text[i - 2], raw_text[i - 1],
#                raw_text[i + 1], raw_text[i + 2]]
#     target = raw_text[i]
#     data.append((context, target))
# print(data[:5])


# def make_context_vector(context, word_to_ix):
#     idxs = [word_to_ix[w] for w in context]
#     return torch.tensor(idxs, dtype=torch.long)
