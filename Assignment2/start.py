import os
import argparse
import pickle as pkl

from torch.utils.data import DataLoader

from utils import data_processing
from word2vec import Word2Vec, Word2VecDataset, train

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

    infrequent_words = data_processing.load_pickle(f"vocabs/infrequent_words_{ARGS.freq_thresh}.pkl")

    print("Loading data ...")
    data = data_processing.get_w2v_data(ARGS)

    vocab = data["vocab"]

    print("Initializing model ...")
    model = Word2Vec(vocab, ARGS.embed_dim).to(ARGS.device)

    print("Initializing dataset")
    word2vec_dataset = Word2VecDataset(data, ARGS)
    data_loader = DataLoader(word2vec_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=2)

    train(ARGS, data_loader, model)
