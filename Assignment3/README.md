# Information Retrieval 1 Homework 3


## Pointwise LTR 
Train the best pointwise LTR model and evaluate it on the test set with 
```python
pointwise_ltr.py 
```
Set different parameters by altering line 196 to 
```python
model, loss_curve, ndcg_val_curve = train_pointwise_ltr(data,
                                                        n_hidden=512, 
                                                        lr=1e-5, 
                                                        batch_size=50)
```
with your respective parameter choices.

## Pairwise LTR

## Listwise LTR with LambdaRank
Train the listwise LTR model with
```python
lambdarank.py [-h] [--epochs EPOCHS] [--n-hidden N_HIDDEN]
                   [--batch-size BATCH_SIZE] [--bpe BPE] [--gamma GAMMA]
                   [--lr LR] [--irm-type IRM_TYPE] [--device DEVICE]
```
Explanations of the parameters can be found in lambdarank.py .
