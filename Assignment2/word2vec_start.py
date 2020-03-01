import os
import json
import argparse
import pickle as pkl

import torch
from torch.utils.data import DataLoader

from utils import data_processing, evaluate
from word2vec import Word2Vec, Word2VecDataset, train, W2VRetrieval, load_model

import read_ap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--save-dir', type=str, default="./word2vec", help="Where outputs are saved")
    parser.add_argument('--find-infreq-words', type=bool, default=False,
                        help="Run function filtering infrequent words")
    parser.add_argument('--start-iter', type=int, default=0, help="Start iteration for w2v data generation.")
    parser.add_argument('--load-data-checkpoint', type=bool, default=False, help="Load checkpoint dict")
    parser.add_argument('--data-load-iter', type=str, default="final", help="Data_{} appendix for loading.")

    # Mode
    parser.add_argument('--mode', type=str, default="train",
                        help="train: training, ret_words: word retrieval (AQ2.1), eval: full evaluation (AQ4.1)"
                             "t-test: perform t-test between model results you load")

    parser.add_argument('--ttest-paths', type=str, default="", nargs='+', help="Two paths to model results")

    # Query
    parser.add_argument('--query', type=str, default="ant", help="Query to be evaluated")
    parser.add_argument('--aggr', type=str, default="mean", help="Type of aggregation to use")
    parser.add_argument('--top-n', type=int, default=11, help="Top N of best matching results")

    # Training
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--save-interval', type=int, default=20000, help='save every save_interval iterations')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    # Word2vec
    parser.add_argument('--ww-size', type=int, default=20, help='Size of word window. Needs to be dividable by 2.')
    parser.add_argument('--embed-dim', type=int, default=300, help='Size of word embedding')

    # Vocabulary
    parser.add_argument('--keep-top-n', type=bool, default=False,
                        help='Keep top-n frequent words. n will be the (arg) --vocab-size')
    parser.add_argument('--freq-thresh', type=int, default=150,
                        help="Filter infrequent words. "
                             "If keep-top-n=True: Keep N top freq words (directly determines vocab-size)"
                             "else: filter words with freq < N")

    ARGS = parser.parse_args()

    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)
        os.makedirs(os.path.join(ARGS.save_dir, "models"))
    if not os.path.exists("filtered_docs/"):
        os.makedirs("filtered_docs/")
    if not os.path.exists("vocabs/"):
        os.makedirs("vocabs/")

    if ARGS.find_infreq_words:
        data_processing.remove_infrequent_words(ARGS)

    vocab = data_processing.load_pickle(f"vocabs/vocab_{ARGS.freq_thresh}.pkl")

    if ARGS.mode == "train":
        print("Starting Training branch...")

        print("Loading data ...")
        data = data_processing.get_w2v_data(ARGS)

        print("Initializing dataset and data loader...")
        word2vec_dataset = Word2VecDataset(data, ARGS)
        data_loader = DataLoader(word2vec_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=2)

        print("Initializing model ...")
        model = Word2Vec(vocab, ARGS.embed_dim).to(ARGS.device)

        print("Train...")
        train(ARGS, data_loader, model)

    elif ARGS.mode in ["ret_words", "eval"]:
        model_path = os.path.join("models", f"ww_{ARGS.ww_size}_{ARGS.freq_thresh}")
        model = load_model(ARGS, model_path)
        print(model)

        print(f"Load docs: filtered_docs/filtered_docs_{ARGS.freq_thresh}.pkl...")
        docs_by_id = data_processing.load_pickle(f"filtered_docs/filtered_docs_{ARGS.freq_thresh}.pkl")
        retriever = W2VRetrieval(ARGS, model, docs_by_id)

        if ARGS.mode == "ret_words":
            print(f"Search query: {ARGS.query}")
            if ARGS.eval_mode == "words":
                top_words = retriever.match_query_against_words(ARGS.query)
                print(f"Top {ARGS.top_n} words:", top_words)

        elif ARGS.mode == "eval":
            qrels, queries = read_ap.read_qrels()
            retriever.evaluate_queries(qrels, queries, model_path)

    # Performs t-tests for exercise 4.2 and 4.4
    elif ARGS.mode == "t-test":
        path1, path2 = ARGS.ttest_paths
        evaluate.ttest_two_models(path1, path2)
