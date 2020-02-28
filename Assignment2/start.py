import os
import argparse
import pickle as pkl

import torch
from torch.utils.data import DataLoader

from utils import data_processing
from word2vec import Word2Vec, Word2VecDataset, train, W2VRetrieval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--save-dir', type=str, default="./word2vec", help="Where outputs are saved")
    parser.add_argument('--find-infreq-words', type=bool, default=False,
                        help="Run function filtering infrequent words")
    parser.add_argument('--freq-thresh', type=int, default=150, help="How much data will be used")
    parser.add_argument('--start-iter', type=int, default=0, help="Start iteration for w2v data generation.")
    parser.add_argument('--load-data-checkpoint', type=bool, default=False, help="Load checkpoint dict")
    parser.add_argument('--data-load-iter', type=str, default="final", help="Data_{} appendix for loading.")

    parser.add_argument('--mode', type=str, default="train", help="Whether perform training or retrieval.")

    # Query
    parser.add_argument('--query', type=str, default="ant", help="Query to be evaluated")
    parser.add_argument('--aggr', type=str, default="mean", help="Type of aggregation to use")
    parser.add_argument('--ret-mode', type=str, default="words", help="Query words (words) or documents (docs)")
    parser.add_argument('--top-n', type=int, default=10, help="Top N of best matching results")

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--save-interval', type=int, default=20000, help='save every save_interval iterations')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    # Word2vec
    parser.add_argument('--ww-size', type=int, default=20, help='Size of word window. Needs to be dividable by 2.')
    parser.add_argument('--embed-dim', type=int, default=100, help='Size of word embedding')

    ARGS = parser.parse_args()

    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)
        os.makedirs(os.path.join(ARGS.save_dir, "models"))

    if ARGS.find_infreq_words:
        data_processing.remove_frequent_words(ARGS.freq_thresh)

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

    elif ARGS.mode == "retrieval":

        print(f"Load model: ww_{ARGS.ww_size}/model_final.pth...")
        model = torch.load(os.path.join(ARGS.save_dir, "models", f"ww_{ARGS.ww_size}", "model_final.pth"))
        model.eval()

        docs_by_id = data_processing.load_pickle(f"filtered_docs/filtered_docs_{ARGS.freq_thresh}.pkl")
        retriever = W2VRetrieval(ARGS, model, docs_by_id)
        print(f"Search query: {ARGS.query}")
        if ARGS.ret_mode == "words":
            top_words = retriever.match_query_against_words(ARGS.query)
            print(f"Top {ARGS.top_n} words:", top_words)
        if ARGS.ret_mode == "docs":
            top_docs = retriever.match_query_against_docs(ARGS.query)
            print(f"Top {ARGS.top_n} docs (by doc id):", top_docs)
