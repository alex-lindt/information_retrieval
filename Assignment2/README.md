# IR1 Homework 2 - UvA

## Prerequisites
Anaconda: https://www.anaconda.com/distribution/

## Getting Started 
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
To train, evaluate and perform inference with the Word2vec model execute the `word2vec_start.py` file with the 
corresponding arguments.
```bash 
    
Usage overview: word2vec_start.py 
    [-h] [--save-dir SAVE_DIR] [--find-infreq-words FIND_INFREQ_WORDS] 
    [--start-iter START_ITER] [--load-data-checkpoint LOAD_DATA_CHECKPOINT] 
    [--data-load-iter DATA_LOAD_ITER] [--mode MODE] 
    [--ttest-paths TTEST_PATHS [TTEST_PATHS ...]] [--query QUERY] 
    [--aggr AGGR] [--top-n TOP_N] [--epochs EPOCHS] [--batch-size BATCH_SIZE] 
    [--lr LR] [--save-interval SAVE_INTERVAL] [--device DEVICE] [--ww-size WW_SIZE] 
    [--embed-dim EMBED_DIM] [--keep-top-n KEEP_TOP_N] [--freq-thresh FREQ_THRESH]


 # General
    --save-dir              Location where outputs are saved. type='str'
    --find-infreq-words     Run function filtering infrequent words. type=bool, 
    --start-iter            Start iteration for w2v data generation type=int
    --load-data-checkpoint  If true load data at a specific iteration during data generation 
                            (for low performance devices) type=bool
    --data-load-iter        Set the desired iteration checkpoint (in the data processing process) from  
                            which the data  should be loaded. (default:"final" for loading final model)

# Mode
    --mode                  Set execution mode: type=str
                             - train: training, 
                             - ret_words: word retrieval (for AQ2.1)
                             - eval: full evaluation (for AQ4.1)
                             - t-test: perform t-test between model results you load

    --ttest-paths           Two paths to model results (.json) type=str, nargs='+'

    # Query
    --query                 Query to be evaluated type=str
    --aggr                  Type of aggregation type=str
    --top-n                 Set N for getting the top N of best matching results 

    # Training
    --epochs                Number of epochs
    --batch-size            Batch size
    --lr                    Learning rate
    --save-interval         Save model every save-interval training iterations
    --device                Training device 'cpu' or 'cuda'

    # Word2vec
    --ww-size               Size of word window
    --embed-dim             Size of word embedding

    # Vocabulary
    General note: 
    We either filter the words with a word frequency below --freq-thresh
    or if --keep-top-n is true, --freq-thresh becomes the vocabulary size
    and we keep the words in the top --freq-thresh frequent words
 
    --keep-top-n           If true, we keep top-n frequent words. n = --freq-thresh => vocab-size
    --freq-thresh          Filter infrequent words
                            - If keep-top-n=True: Keep --freq-thresh top freq words
                            - else: filter words with frequency <= --freq-thresh

```

### Doc2Vec
Training and evaluating the default setting model:
`python doc2vec.py `

If you want to change the model settings for training and evaluating a single model, just set them in the following line:
```bash
# (1) Run with default parameters
doc2vec_run_and_evaluate(train_data, vector_size=400, window=8, max_vocab_size=None)
```
If you want to run the grid search, just comment in the following line:
```bash
# (2) Run Grid Search
# run_grid_search(doc_path)
```



### LSI and LDA
The jupyter notebook `LSI_LDA.ipynb` contains all the code needed to train and evaluate the LSI BOW, LSI TFIDF and LDA BOW models. Also it contains the code for the grid search over different topic numbers for LSI BOW and LSI TFIDF. Instructions for running the code are given in the notebook itself. 

## Authors

- Alexandra Lindt (12230642)- alexandra.lindt@student.uva.nl
- David Biertimpel (12324418)- david.biertimpel@student.uva.nl
- Vanessa Botha (10754954) - vanessa.botha@student.uva.nl
