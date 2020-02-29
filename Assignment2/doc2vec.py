import os
import re
import json
import pytrec_eval
import numpy as np
from tqdm import tqdm
import pickle as pkl

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import read_ap

class Logger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.batch = 0

    def on_epoch_begin(self, model):
        print(f"--- EPOCH {self.epoch}")

    def on_batch_begin(self, model):
        if self.batch % 400 == 0 and self.batch != 0:
            print(f"[Batch {self.batch} / 4400]")
        self.batch += 1

    def on_epoch_end(self, model):
        self.batch = 0
        self.epoch += 1

def get_train_data(doc_path):
    assert os.path.isfile(doc_path)

    # load pre-processed text
    # i.e. stemmed / tokenized / stop words removed / rarest 150 removed
    with open(doc_path, "rb") as reader:
        docs_by_id = pkl.load(reader)

    # 'id' -> ['token0', 'token21', '...]
    print("Create Training Data")
    train_data = [TaggedDocument(doc, [doc_id]) for doc_id, doc in docs_by_id.items()]
    print(f"Lenght of Training Data: {len(train_data)}")
    return train_data

def train_doc2vec(train_data, epochs, window, vector_size, max_vocab_size):
    print("#" * 20)
    print("Train Model")
    description = f"E{epochs}_W{window}_VES{vector_size}_MAXVOS{max_vocab_size}"
    print(description)

    model = Doc2Vec(train_data,
                    vector_size=vector_size,
                    min_count=1,
                    epochs=epochs,
                    dm=0,
                    window=window,
                    compute_loss=True,
                    max_vocab_size=max_vocab_size,
                    callbacks=[Logger()])

    print("\nSANITY CHECK")
    print(f"Vocabulary Size: {len(model.wv.vocab)}")
    print(f"Corpus Size: {model.corpus_count}")
    print("#" * 20)

    model.save('./doc2vec/models/doc2vec_' + description)

    return model, description

def evaluate_doc2vec(doc2vec_model, description, test_subset=False):

    qrels, queries = read_ap.read_qrels()

    if test_subset:
        queries = {qid: q for qid, q in queries.items() if int(qid) < 101 and int(qid) > 75}

    overall_ser = {}

    # collect results
    for qid in queries:
        query_text = queries[qid]
        query_repr = read_ap.process_text(query_text)
        query_vector = doc2vec_model.infer_vector(query_repr)

        results = doc2vec_model.docvecs.most_similar([query_vector],
                                                     topn=len(doc2vec_model.docvecs))

        overall_ser[qid] = dict(results)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    if not test_subset:
        with open(f"./doc2vec/results/doc2vec_{description}.json", "w") as writer:
            json.dump(metrics, writer, indent=1)

    return metrics

def run_grid_search(doc_path):
    train_data = get_train_data(doc_path)

    for vector_size in [200, 300, 400, 500]:
        for window in [5,10, 15, 20]:
            for max_vocab_size in np.array([10, 25, 50, 100, 200]) * 1000:
                doc2vec_run_and_evaluate(train_data, vector_size, window, max_vocab_size)

def doc2vec_run_and_evaluate(train_data, vector_size, window, max_vocab_size):

    model, description = train_doc2vec(train_data,
                                       epochs=2,
                                       window=window,
                                       vector_size=vector_size,
                                       max_vocab_size=max_vocab_size)

    model.delete_temporary_training_data()

    metrics = evaluate_doc2vec(model, description, test_subset=False)
    map_all = np.average([m['map'] for m in metrics.values()])
    ndcg_all = np.average([m['ndcg'] for m in metrics.values()])

    metrics = evaluate_doc2vec(model, description, test_subset=True)
    map_subset = np.average([m['map'] for m in metrics.values()])
    ndcg_subset = np.average([m['ndcg'] for m in metrics.values()])

    print(f"\n### EVALUATING :{description}")
    print(f"All    : MAP {map_all}, NDCG {ndcg_all}")
    print(f"76-100 : MAP {map_subset}, NDCG {ndcg_subset}")


def read_grid_search_results(path='./doc2vec/results/doc2vec_gridsearch_results.txt'):
    with open(path) as file:
        text = file.readlines()
    text = [t for t in text if ("###" in t or "All" in t or "76-" in t)]

    # VES - W - MAXVOCAB
    # all queries
    res = {ves: {w: {vs: 0 for vs in [10000, 25000, 50000, 100000, 200000]}
                 for w in [5, 10, 15, 20]}
           for ves in [200, 300, 400, 500]}

    for i in range(len(text)):

        if i % 3 == 0:
            r = {'map_a': 0, 'ndcg_a': 0, 'map_s': 0, 'ndcg_s': 0}

            # get parameters
            line1 = text[i].split(':')[1].split('_')
            [w, ves, vs] = [int(re.sub("[^0-9]", "", t)) for t in line1][-3:]

            # get results for subsets of queries
            res[ves][w][vs] = round(float(text[i + 2].split(' ')[3][:-1]), 4)

    return res

def sort_highest(res_dict, n=5):
    res = []

    for ves in [200, 300, 400, 500]:
        for w in [5, 10, 15, 20]:
            for vs in [10000, 25000, 50000, 100000, 200000]:
                if res_dict[ves][w][vs]:
                    res.append(((ves, w, vs), res_dict[ves][w][vs]))

    res = sorted(res, key=lambda x: -x[1])

    return res[:n]


if __name__ == "__main__":


    doc_path = "processed_docs.pkl"
    train_data = get_train_data(doc_path)

    # (1) Run with default parameters
    doc2vec_run_and_evaluate(train_data,vector_size=400, window=8, max_vocab_size=None)

    # (2) Run Grid Search
    # run_grid_search(doc_path)

    # (3) Process Grid Search Results
    res_dict = read_grid_search_results()
    print(sort_highest(res_dict, n=5))
