# IR1 Homework 2 - UvA

## Prerequisites
Anaconda: https://www.anaconda.com/distribution/

## Getting Started 
### CHECK WHETHER WE NEED TO ADD ANYTHING TO environment.yml
Open the Anaconda prompt and move to this directory

First create and activate the provided environment by using the following commands:
```bash
conda env create -f environment.yml
conda activate ir1-hw2
```

And in order to deactivate the environment:
```bash
conda deactivate
```

## Running the models 
### Word2vec
### Doc2Vec
### LSI and LDA
`LSI_LDA.py` contains the code to train and evaluate the LSI and LDA models.
```bash 
usage: LSI_LDA.py [-h] [--save-dir SAVE_DIR] [--num_topics NUM_TOPICS]
                  [--filter FILTER] [--model_type MODEL_TYPE]
                  [--corpus_type CORPUS_TYPE] [--load_model LOAD_MODEL]
                  [--path_model PATH_MODEL]
                  [--load_tfidfmodel LOAD_TFIDFMODEL]
                  [--path_tfidfmodel PATH_TFIDFMODEL]
                  [--load_corpus LOAD_CORPUS] [--path_corpus PATH_CORPUS]
                  [--load_dict LOAD_DICT] [--path_dict PATH_DICT]

Arguments: 

-h                Shows file usage

--save-dir        Where outputs are saved
--num_topics      Number of topics to find
--model_type      Either LSI or LSA
--corpus_type     Either BOW or TFIDF

--load_model      Load LSI/LDA trained model
--path_model      Path to saved LSI/LDA model
--load_tfidfmodel Load trained tfidf model
--path_tfidfmodel Path to saved tfidf model
--load_corpus     Load corpus
--path_corpus     Path to saved corpus
--load_dict       Load dictionary
--path_dict       Path to saved dict
```

#### Train and evaluate: 
* *LSI BoW:*
`python LSI_LDA.py --model_type LSI --corpus_type BOW `

* *LSI TF-IDF:*
`python LSI_LDA.py --model_type LSI --corpus_type TFIDF`

* *LDA TF-IDF:*
`python LSI_LDA.py --model_type LDA --corpus_type TFIDF`

* The number of topics for training can be set by  e.g. `--num_topics 500`

* Earlier saved corpora and/or models can be loaded using the load and path arguments. Note that if the tf-idf corpus is loaded, the corresponding tf-idf model should be provided as it is needed during evaluation time. 


## Authors

- Alexandra Lindt (12230642)- alexandra.lindt@student.uva.nl
- David Biertimpel (12324418)- david.biertimpel@student.uva.nl
- Vanessa Botha (10754954) - vanessa.botha@student.uva.nl
