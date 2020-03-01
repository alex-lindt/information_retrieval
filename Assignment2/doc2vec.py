import os
import re
import json
import pytrec_eval
import numpy as np
from tqdm import tqdm
import pickle as pkl

from utils import evaluate

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
    # i.e. stemmed / tokenized / stop words removed
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

def rank_query_given_document(query_text, doc2vec_model):
    #   Function that ranks documents given a query
    query_repr = read_ap.process_text(query_text)
    query_vector = doc2vec_model.infer_vector(query_repr)

    results = doc2vec_model.docvecs.most_similar([query_vector],
                                                 topn=len(doc2vec_model.docvecs))
    return results

def evaluate_doc2vec(doc2vec_model, description, test_subset=False):

    qrels, queries = read_ap.read_qrels()

    if test_subset:
        queries = {qid: q for qid, q in queries.items() if int(qid) < 101 and int(qid) > 75}

    overall_ser = {}
    # collect results
    for qid in queries:
        results = rank_query_given_document(queries[qid], doc2vec_model)
        overall_ser[qid] = dict(results)

        if int(qid) not in np.arange(76, 101):
            evaluate.write_trec_results(qid, results, f"./doc2vec/results/")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    if not test_subset:
        with open(f"./doc2vec/results/doc2vec_{description}.json", "w") as writer:
            json.dump(metrics, writer, indent=1)

    return metrics

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

    map_subset = np.average([m['map'] for qid, m in metrics.items()
                             if int(qid) in range(76,101)])

    ndcg_subset = np.average([m['map'] for qid, m in metrics.items()
                              if int(qid) in range(76,101)])

    print(f"\n### EVALUATING :{description}")
    print(f"All    : MAP {map_all}, NDCG {ndcg_all}")
    print(f"76-100 : MAP {map_subset}, NDCG {ndcg_subset}")

def run_grid_search(doc_path):
    train_data = get_train_data(doc_path)

    for vector_size in [200, 300, 400, 500]:
        for window in [5, 10, 15, 20]:
            for max_vocab_size in np.array([10, 25, 50, 100, 200]) * 1000:
                doc2vec_run_and_evaluate(train_data, vector_size, window, max_vocab_size)


if __name__ == "__main__":

    if not os.path.exists("./doc2vec"):
        os.mkdir("./doc2vec")
        os.mkdir("./doc2vec/results")
        os.mkdir("./doc2vec/models")

    train_data = get_train_data("processed_docs.pkl")

    # (1) Run with default parameters
    doc2vec_run_and_evaluate(train_data, vector_size=400, window=8, max_vocab_size=None)

    # (2) Run Grid Search
    # run_grid_search(doc_path)

    # (3) Process Grid Search Results
    # res_dict = read_grid_search_results()
    # print(sort_highest(res_dict, n=5))
