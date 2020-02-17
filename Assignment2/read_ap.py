import string
import os
import pickle as pkl
from os import listdir
from os.path import isfile, join
import sys
import time

import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import word_tokenize

from multiprocessing import Pool


nltk.download("stopwords")
nltk.download('punkt')
eng_stopwords = set(stopwords.words('english')).union(set(string.punctuation))

tokenizer = TreebankWordTokenizer()


def stem_token(token):
    """
        Stem the given token, using any stemmer available from the nltk library
        Input: a single token
        Output: the stem of the token
    """
    from nltk.stem.porter import PorterStemmer

    return PorterStemmer().stem(token)


def tokenize(text):
    """
        Tokenize the text.
        Input: text - a string
        Output: a list of tokens
    """
    tokens = word_tokenize(text)
    return tokens


def process_text(text):
    tokens = []
    for token in tokenize(text):
        if token.lower() in eng_stopwords:
            continue
        token = stem_token(token)
        token = token.lower()
        tokens.append(token)

    return tokens


def read_ap_docs(root_folder="./datasets/"):
    dirs = [join(root_folder, "ap", "docs", 'ap-88'),
            join(root_folder, "ap", "docs", 'ap-89')]
    doc_ids = []
    docs = []

    apfiles = []
    for dir in dirs:
        apfiles.extend([join(dir, f) for f in listdir(dir) if isfile(
            join(dir, f)) and 'ap' in f])

    print("Reading in documents")
    for apfile in tqdm(apfiles):
        with open(apfile, 'r', errors='replace') as reader:
            lines = reader.readlines()
        line_counter = 0
        doc_id = ''
        doc = ''
        while line_counter < len(lines):
            line = lines[line_counter]
            if '<DOCNO>' in line:
                doc_id = line.split('<DOCNO>')[1].strip().split(
                    '</DOCNO>')[0].strip()
                doc = ''
                doc_ids.append(doc_id)
            if '<TEXT>' in line and '</TEXT>' not in line:
                line_counter += 1
                line = lines[line_counter]
                while '</TEXT>' not in line:
                    doc += line.strip() + " "
                    line_counter += 1
                    line = lines[line_counter]
                if len(docs) == len(doc_ids):
                    docs[-1] = doc
                else:
                    docs.append(doc)
                continue
            line_counter += 1

    return docs, doc_ids


def get_processed_docs(doc_set_name="processed_docs"):

    path = f"./{doc_set_name}.pkl"

    if not os.path.exists(path):
        docs, doc_ids = read_ap_docs()

        print("Processing documents now")
        doc_repr = {}
        p = Pool()
        out_p = []
        step_size = 1000
        start_time = time.time()
        for i in range(0, len(docs), step_size):
            out_p_local = p.map(
                process_text, docs[i:min(len(docs), i+step_size)])
            out_p += out_p_local
            print("Processed %i of %i docs" % (i+step_size, len(docs)))
            time_passed = time.time() - start_time
            time_to_go = time_passed * (len(docs)-i-step_size) / (i+step_size)
            print("Estimated remaining time: %imin %isec" %
                  (int(time_to_go/60.0), int(time_to_go) % 60))

        for i in range(len(out_p)):
            if len(out_p[i]) > 0:
                doc_repr[doc_ids[i]] = out_p[i]

        with open(path, "wb") as writer:
            pkl.dump(doc_repr, writer)

        print(f"all docs processed. saved to {path}")

        return doc_repr
    else:
        print("Docs already processed. Loading from disk")

        with open(path, "rb") as reader:
            return pkl.load(reader)


def read_qrels(root_folder="./datasets/"):

    qrels = {}
    queries = {}

    with open(os.path.join(root_folder, "ap", "qrels.tsv")) as reader:
        for line in reader:
            qid, _, doc_id, _ = line.split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = 1

    with open(os.path.join(root_folder, "ap", "queries.tsv")) as reader:
        for line in reader:
            qid, query = line.split("\t")
            if qid in qrels:
                queries[qid] = query

    return qrels, queries


if __name__ == "__main__":
    get_processed_docs()
