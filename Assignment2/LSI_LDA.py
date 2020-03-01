from gensim.models import LsiModel, LdaModel, CoherenceModel, TfidfModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import kullback_leibler
from gensim import similarities

from collections import defaultdict
from tqdm import tqdm

import argparse
import json
import os 
# import pytrec
import time


import read_ap
import download_ap



def get_data():
    print("Loading data ...") 

    # load preprocessed data 
    download_ap.download_dataset()
    docs_by_id = read_ap.get_processed_docs()
    
    return docs_by_id

def get_dict(docs):
    print("Building dictionary ...")
    if ARGS.load_dict: 
        # load dictionary from disk
        dictionary = Dictionary.load(ARGS.path_dict)
    else: 
        # create dictionary and save to disk 
        dictionary = Dictionary(docs)

        if ARGS.filter:
            print("Filter extremes")
            dictionary.filter_extremes(no_below=25, no_above=0.5)

        dictionary.save(ARGS.save_dir + '/corpora/dictionary.dict')  
    return dictionary


def get_corpus(docs):
    print("Building corpus ...")
    tfidf_model = None

    # load corpus from disk 
    if ARGS.load_corpus: 
        corpus = MmCorpus(ARGS.path_corpus)

    else:
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        # serialize corpus to disk to prevent memory problems if corpus gets too large
        MmCorpus.serialize(ARGS.save_dir + '/corpora/corpus_bow.mm', corpus)  
        corpus = MmCorpus(ARGS.save_dir + '/corpora/corpus_bow.mm')

        if ARGS.corpus_type == "TFIDF": 
            tfidf_model = TfidfModel(corpus)

            tfidf_model.save(ARGS.save_dir + "/models/tfidf_model.mm")
            corpus = tfidf_model[corpus]

            # serialize corpus to disk to prevent memory problems if corpus gets too large
            MmCorpus.serialize(ARGS.save_dir + '/corpora/corpus_tfidf.mm', corpus)  
            corpus = MmCorpus(ARGS.save_dir + '/corpora/corpus_tfidf.mm')
    return corpus, tfidf_model


def train(corpus, dictionary):
    print("Training model ...")
    print("Number of topics:", ARGS.num_topics)

    if ARGS.model_type == "LSI": 
        model = LsiModel(corpus, id2word=dictionary, num_topics=ARGS.num_topics)
        model.save(ARGS.save_dir + "/models/"+ARGS.model_type+"_"+ARGS.corpus_type+".mm")

    elif ARGS.model_type == "LDA": 
        model = LdaModel(corpus, id2word=dictionary, num_topics=ARGS.num_topics)
        model.save(ARGS.save_dir + "/models/"+ARGS.model_type+"_"+ARGS.corpus_type+".mm")

    return model 


########################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#### NEEDS TO BE UPDATED IN THE END ############
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
########################################################
def rank_docs(query, model, doc_ids, dictionary, corpus_modelspace, tfidf_model=None, index=None):
    query_prepro = read_ap.process_text(query)

    # transform query to bow vector space
    q_cspace = dictionary.doc2bow(query_prepro)

    if not tfidf_model == None:
        # transform query to tfidf vector space
        q_cspace = tfidf_model[q_cspace]

    q_modelspace = model[q_cspace]
    
    if isinstance(model, LsiModel):
        print("lsi")
        scores = index[q_modelspace] # TODO this should be the absolute value

    elif isinstance(model, LdaModel):
      print("lda")
      # TODO: cast to float for pytrec
      scores = - kullback_leibler(q_modelspace, corpus_modelspace)

    results = defaultdict(float)
    for doc_id, score in zip(doc_ids, scores):
      results[doc_id] = score

    results = list(results.items())
    results.sort(key=lambda _: -_[1])
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--save-dir', type=str, default="./LSI_LDA", help="Where outputs are saved")

    # Params
    parser.add_argument('--num_topics', type=int, default=500, help="Number of topics to find")
    parser.add_argument('--filter', type=bool, default=True, help="Filter tokens with extreme occurences")
    parser.add_argument('--model_type', type=str, default="LSI", help="Either LSI or LDA")
    parser.add_argument('--corpus_type', type=str, default="BOW", help="Either BOW or TFIDF")

    # Load models 
    parser.add_argument('--load_model', type=bool, default=False, help="Load LSI/LDA trained model")
    parser.add_argument('--path_model', type=str, default="./LSI_LDA/models/LSI_BOW.mm", help="Path to saved LSI/LDA model")

    parser.add_argument('--load_tfidfmodel', type=bool, default=False, help="Load trained tfidf model")
    parser.add_argument('--path_tfidfmodel', type=str, default="./LSI_LDA/models/tfidf_model.mm", help="Path to saved tfidf model")

    # Load corpora
    parser.add_argument('--load_corpus', type=bool, default=False, help="Load corpus")
    parser.add_argument('--path_corpus', type=str, default="./LSI_LDA/corpora/corpus_bow.mm", help="Path to saved corpus")
    parser.add_argument('--load_dict', type=bool, default=False, help="Load dictionary")
    parser.add_argument('--path_dict', type=str, default="./LSI_LDA/corpora/dictionary.dict", help="Path to saved dict")

    

    ARGS = parser.parse_args()

    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)
        os.makedirs(os.path.join(ARGS.save_dir, "models"))
        os.makedirs(os.path.join(ARGS.save_dir, "corpora"))

    docs_by_id = get_data()

    # test on subset
    docs_by_id = dict(list(docs_by_id.items())[:10])

    docs = docs_by_id.values()
    
    dictionary = get_dict(docs)
    

    tfidf_model = None
    corpus, tfidf_model = get_corpus(docs)
    

    if ARGS.corpus_type == "TFIDF" and tfidf_model == None:
        if ARGS.load_tfidfmodel:
            tfidf_model = TfidfModel.load(ARGS.path_tfidfmodel)
        else:
            raise Exception("TFIDF model should be loaded")

    if ARGS.load_model:
        # load model from disk if path is given 
        if ARGS.model_type == "LSI": 
            model = LsiModel.load(ARGS.path_model)
        elif ARGS.model_type == "LDA": 
            model = LdaModel.load(ARGS.path_model)
        else:
            raise Exception("Unsupported model type")
    else:
        # train model 
        model = train(corpus, dictionary)
        

    top5_topics = dict([(int(topic_nr), tokens.split("+")) for topic_nr, tokens in model.print_topics(num_topics=5, num_words=20)])
    print("Top 5 most significant topics:")
    print(top5_topics)

    with open(ARGS.save_dir + "/" + ARGS.model_type + "_" + ARGS.corpus_type + "_top5_topics.json", "w") as writer:
        json.dump(top5_topics, writer, indent=1)




    ##### TEST THIS LAST PART ########### 


    # # transform corpus to model space
    # corpus_modelspace = model[corpus]

    # # only needed for evaluation of LSI model 
    # index = similarities.MatrixSimilarity(corpus_modelspace, dtype=float) if ARGS.model_type == "LSI" else None

    # # evaluate on the queries
    # qrels, queries = read_ap.read_qrels()

    # overall_result = {}

    # for query_id, query in tqdm(queries.items()): 
    #   results = rank_docs(query, model, docs_by_id.keys(), dictionary, corpus_modelspace, tfidf_model=tfidf_model, index=index)
    #   overall_result[query_id] = dict(results)


    # evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    # metrics = evaluator.evaluate(overall_result)

    # print("Mean MAP: ", np.average([m['map'] for m in metrics.values()]))
    # print("Mean NDCG: ", np.average([m['ndcg'] for m in metrics.values()]))

    # # dump this to JSON
    # with open(ARGS.model_type+"_"+ARGS.corpus_type+".json", "w") as writer:
    #     json.dump(metrics, writer, indent=1)

    




