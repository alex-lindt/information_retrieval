import os
import random
import pickle as pkl
from collections import Counter

import numpy as np

from gensim.corpora import Dictionary
from six import PY3, iteritems, iterkeys, itervalues, string_types

from tqdm import tqdm

import read_ap
import download_ap

from gensim.parsing.preprocessing import remove_stopwords


def get_w2v_data(ARGS):
    data_path = os.path.join(ARGS.save_dir, "data_{}")
    if os.path.exists(data_path):
        with open(data_path, "rb") as reader:
            data = pkl.load(reader)
            if data["ww_size"] != ARGS.ww_size:
                print(f"Error! word window size of data ({data['ww_size']}) does not match the input ({ARGS.ww_size})")
                # Could terminate here ...
    else:
        with open("infrequent_words.pkl", "rb") as reader:
            infrequent_words = pkl.load(reader)
        data = build_w2v_data(data_path, infrequent_words, ARGS)

    return data


def load_docs(path_name):
    with open(path_name, "rb") as reader:
        data = pkl.load(reader)
    return data


def build_w2v_data(data_path, vocabulary, ARGS):
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = load_docs("filtered_docs.pkl")

    vocab_size = len(vocabulary)

    print("Number of documents", len(docs_by_id))
    print("Size vocabulary:", vocab_size)

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

        # we don't consider the first ww_size words as it doesn't make a difference in the limit
        for i_target in range(ww_size, doc_length - ww_size):
            word_context = []

            word_context.extend(doc[i_target - ww_size:i_target])
            word_context.extend(doc[i_target + 1:i_target + ww_size + 1])

            target = doc[i_target]

            # negative sampling
            k = ww_size * 2
            _samples = np.random.randint(vocab_size, size=k)

            negative_sample = []
            for s in _samples:
                negative_sample.append(vocabulary.id2token[s])

            data["target"].append(target)
            data["context"].append(word_context)
            data["negatives"].append(negative_sample)  # does it make a difference that we don't do this in the end?

        if (i+1) % 40000 == 0:
            data["vocab"] = vocabulary

            with open(data_path.format(i), "wb") as writer:
                pkl.dump(data, writer)

    data["vocab"] = vocabulary

    with open(data_path.format("final"), "wb") as writer:
        pkl.dump(data, writer)

    return data


def find_frequent_words(freq_thresh):
    print("Find infrequent words...")
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    word_dict = Dictionary(docs_by_id.values())
    word_dict.filter_n_most_frequent(remove_n=freq_thresh)
    # infrequent_words = set(word_dict.token2id.keys())

    word_dict.id2token = {v: k for k, v in word_dict.token2id.items()}

    print(word_dict.id2token)
    print(word_dict.id2token.keys())

    with open(f"./infrequent_words.pkl", "wb") as writer:
        pkl.dump(word_dict, writer)


def remove_frequent_words():
    print("Remove infrequent words...")
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    with open("infrequent_words.pkl", "rb") as reader:
        infrequent_words = pkl.load(reader)

    new_docs = {}
    for i, (doc_id, _doc) in tqdm(enumerate(docs_by_id.items())):
        doc = [w for w in _doc if w in infrequent_words.token2id]
        new_docs[doc_id] = doc

    with open(f"./filtered_docs.pkl", "wb") as writer:
        pkl.dump(new_docs, writer)
