import os
import random
import pickle as pkl
from collections import Counter

import itertools

import numpy as np

from gensim.corpora import Dictionary
from six import PY3, iteritems, iterkeys, itervalues, string_types

from tqdm import tqdm

import read_ap
import download_ap

from gensim.parsing.preprocessing import remove_stopwords


def get_w2v_data(ARGS):
    data_path = os.path.join(ARGS.save_dir, "data_{}")
    if os.path.exists(data_path.format(ARGS.data_load_iter)):
        return load_pickle(data_path.format(ARGS.data_load_iter))
    else:
        infrequent_words = load_pickle(f"vocabs/infrequent_words_{ARGS.freq_thresh}.pkl")
        return build_w2v_data(data_path, infrequent_words, ARGS, start_iter=ARGS.start_iter)


def build_w2v_data(data_path, vocabulary, ARGS, start_iter=0):
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = load_pickle(f"filtered_docs/filtered_docs_{ARGS.freq_thresh}.pkl")

    vocab_size = len(vocabulary)

    id2token = {v: k for k, v in vocabulary.items()}
    full_vocab = {
        "id2token": id2token,
        "token2id": vocabulary
    }

    print("Number of documents", len(list(docs_by_id.items())[start_iter:]))
    print("Size vocabulary:", vocab_size)

    ww_size = ARGS.ww_size
    ww_size_half = int(ww_size / 2)

    if ARGS.load_data_checkpoint:
        data = load_pickle(data_path.format(start_iter))

    else:
        data = {
            "target": [],
            "context": [],
            "negatives": [],
            "ww_size": ww_size,
            "vocab": full_vocab
        }

    # Create instance for retrieval
    for i, (doc_id, doc) in tqdm(enumerate(list(docs_by_id.items())[start_iter:])):

        doc_length = len(doc)

        # we don't consider the first ww_size words as it doesn't make a difference in the limit
        for i_target in range(ww_size, doc_length - ww_size):

            word_context = []
            word_context.extend(doc[i_target - ww_size_half:i_target])
            word_context.extend(doc[i_target + 1:i_target + ww_size_half + 1])

            target = doc[i_target]

            # negative sampling
            k = ww_size
            # _samples = np.random.randint(vocab_size, size=k)
            _samples = [int(vocab_size * random.random()) for i in range(k)]
            negative_sample = []
            # Although a for loop, way more efficient than random.sample
            for s in _samples:
                negative_sample.append(full_vocab["id2token"][s])

            data["target"].append(target)
            data["context"].append(word_context)
            data["negatives"].append(negative_sample)  # does it make a difference that we don't do this in the end?

        if i >= 60000:
            break

        # if (start_iter + i + 1) % 80000 == 0:
        #     save_pickle(data, data_path.format(start_iter + i))

    save_pickle(data, data_path.format("final"))

    return data


def find_frequent_words(freq_thresh):
    print("Find infrequent words...")
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    total_counts = Counter(itertools.chain.from_iterable(docs_by_id.values()))
    infrequent_words = Counter(el for el in total_counts.elements() if total_counts[el] >= freq_thresh)

    # word_dict = Dictionary(docs_by_id.values())
    # word_dict.filter_n_most_frequent(remove_n=freq_thresh)
    # infrequent_words = set(word_dict.token2id.keys())

    save_pickle(infrequent_words, f"infrequent_words_{freq_thresh}.pkl")


def remove_frequent_words(freq_thresh):
    print("Remove infrequent words...")
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # infrequent_words = load_pickle(f"infrequent_words_{freq}.pkl")

    total_counts = Counter(itertools.chain.from_iterable(docs_by_id.values()))
    infrequent_words = Counter(el for el in total_counts.elements() if total_counts[el] >= freq_thresh)
    token2id = {w: i for i, w in enumerate(infrequent_words)}

    new_docs = {}
    for i, (doc_id, _doc) in tqdm(enumerate(docs_by_id.items())):
        doc = [w for w in _doc if w in infrequent_words]
        new_docs[doc_id] = doc

    if not os.path.exists("filtered_docs/"):
        os.makedirs("filtered_docs/")
    if not os.path.exists("vocabs/"):
        os.makedirs("vocabs/")

    save_pickle(token2id, f"vocabs/infrequent_words_{freq_thresh}.pkl")
    save_pickle(new_docs, f"filtered_docs/filtered_docs_{freq_thresh}.pkl")


def load_pickle(path_name):
    with open(path_name, "rb") as reader:
        data = pkl.load(reader)
    return data


def save_pickle(data, path_name):
    with open(path_name, "wb") as writer:
        pkl.dump(data, writer)
