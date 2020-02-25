import os
import random
import pickle as pkl
from collections import Counter

from tqdm import tqdm

import read_ap
import download_ap


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
            word_context.extend(doc[i_target + 1:i_target + ww_size + 1])

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


def filter_infrequent_words(use_data=20000):
    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    infreq_list = []
    general_counts = Counter()

    # filter frequent words
    for i, (doc_id, doc) in tqdm(enumerate(docs_by_id.items())):
        if i >= use_data:
            break

        counts = Counter(doc)
        general_counts += counts

    for t, c in tqdm(general_counts.items()):
        if c <= 50:
            infreq_list.append(t)

    filtered_docs_by_id = {}
    for i, (doc_id, doc) in tqdm(enumerate(docs_by_id.items())):
        if i >= use_data:
            break
        _removed = [word for word in doc if word not in infreq_list]
        filtered_docs_by_id[doc_id] = _removed

    with open(f"./processed_docs_filtered.pkl", "wb") as writer:
        pkl.dump(filtered_docs_by_id, writer)
