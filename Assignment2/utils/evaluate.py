import os

import numpy as np

import pytrec_eval
import scipy.stats

from utils import data_processing


def ttest_two_models(path1, path2):
    results1 = data_processing.load_json(path1)
    results2 = data_processing.load_json(path2)

    ttest_results = [(test_significance(results1, results2, metric), metric) for metric in ['map', 'ndcg']]

    for res, metric in ttest_results:
        print("-----" * 10)
        print(f"Metric: {metric}, Result: {res}")
        print("-----" * 10)


#   TODO - Ttest combos:
#          TF-IDF - Word2Vec
#          TF-IDF - Doc2Vec
#          TF-IDF - LSIBow
#          TF-IDF - LSI-TF-IDF
#          Word2Vec - Doc2Vec
#          Word2Vec - LSIBow
#          Word2Vec - LSI-TF-IDF
#          Doc2Vec - LSIBow
#          Doc2Vec - LSI-TF-IDF
#          LSIBow - LSI-TF-IDF


def test_significance(results_1, results_2, metric):
    query_ids = list(set(results_1.keys()) & set(results_2.keys()))

    first_scores = [results_1[query_id][metric] for query_id in query_ids]
    second_scores = [results_2[query_id][metric] for query_id in query_ids]

    return scipy.stats.ttest_rel(first_scores, second_scores)


def calc_mean_metrics(metrics_dict):
    maps_all, ndcg_all = [], []
    maps_valid, ndcg_valid = [], []
    for id, query in metrics_dict.items():
        map = query["map"]
        ndcg = query["ndcg"]
        maps_all.append(map)
        ndcg_all.append(ndcg)
        if int(id) in np.arange(76, 101):
            maps_valid.append(map)
            ndcg_valid.append(ndcg)

    print("----" * 10)
    print(f"Mean MAP (all): {np.mean(maps_all)}")
    print(f"Mean NDCG (all): {np.mean(ndcg_all)}")
    print("----" * 10)
    print(f"Mean MAP (76-100): {np.mean(maps_valid)}")
    print(f"Mean NDCG (76-100): {np.mean(ndcg_valid)}")
    print("----" * 10)


def write_trec_results(qid, results, path):
    # query-id Q0 document-id rank score STANDARD
    with open(os.path.join(path, 'trec_results.txt'), 'a') as the_file:
        for rank, (doc_id, score) in enumerate(results):
            # Only the top 1000 otherwise the file blow up
            if rank > 1000:
                break
            the_file.write(f'{qid} Q0 {doc_id} {rank + 1} {score} STANDARD \n')

# def write_result_file():
#     with open('somefile.txt', 'a') as the_file:
#         for i in range(150 * 160000):
#             the_file.write('query-id Q0 document-id rank score STANDARD \n')
#
#     with open('somefile1.txt', 'a') as the_file:
#         for i in range(50 * 160000):
#             the_file.write('query-id Q0 document-id rank score STANDARD \n')
#
