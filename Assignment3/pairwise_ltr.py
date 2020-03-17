import dataset
import torch
from torch import nn
from pointwise_ltr import sample_batch
import numpy as np
from itertools import permutations 
import matplotlib.pyplot as plt
import evaluate as evl

from pointwise_ltr import progress_over_last

from tqdm import tqdm

class Pairwise_LTR_Model(nn.Module):

    def __init__(self, device, n_hidden):    
    
        super().__init__()

        self.input_size = 501
        self.output_size = 1
        
        self.nn = torch.nn.Sequential(
            nn.Linear(self.input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,self.output_size)
        )

        self.to(device)
    
    def forward(self, x):
        return self.nn(x)


class RankNetLoss(nn.Module):
    def __init__(self, sigma, batch_size):  

        super().__init__()

        self.sigma = sigma
        self.batch_size = batch_size

    def _S(self, rel_i, rel_j): 
        diff = rel_i - rel_j

        if diff == 0:
            return 0
        else:
            # returns -1 if diff is negative and +1 if diff is positive
            return diff ** 0 


    def forward(self, pred_score, labels):

        pairs_all = list(permutations(np.arange(len(labels)), 2))

        np.random.shuffle(pairs_all)

        # select only a fixed subset of pairs to consider for the loss 
        pairs_batch = pairs_all[:self.batch_size]

        C_T = 0 

        for doc_i, doc_j in pairs_all: 

            C = (0.5 * (1 - self._S(labels[doc_i], labels[doc_j]))
                    * self.sigma * (pred_score[doc_i] - pred_score[doc_j]) 
                    + torch.log2(1 + torch.exp(-self.sigma*(pred_score[doc_i] - pred_score[doc_j]))))

            C_T += C

        return C_T / len(pairs_all)

            

def get_samples_by_qid(qid, data_split, device):

    docs = data_split.query_feat(qid)
    labels = data_split.query_labels(qid)

    return torch.Tensor(docs).to(device), torch.Tensor(labels).to(device) 


def train_ranknet(data, epochs=100, n_hidden=1024, lr=1e-4, batch_size=10, spedup = True, device='cpu'):


    model = Pairwise_LTR_Model(device, n_hidden=n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    if spedup:
        pass # implement sped up ranknetloss here
    else:
        loss_fn = RankNetLoss(sigma=1.0, batch_size=batch_size)

    # track loss and ndcg on validation set 
    loss_curve = []
    ndcg_val_curve = []


    queries = np.arange(0, data.train.num_queries())

    for epoch in range(epochs): 

        loss_epoch = []
        np.random.shuffle(queries)
     
        for qid in tqdm(queries): 

            docs, labels = get_samples_by_qid(qid=qid, data_split=data.train, device=device)

            # if there is only one doc retrieved by the query
            if docs.shape[0] == 1: 
                continue 

            pred_score = model.forward(docs)

            loss = loss_fn(pred_score, labels)
            loss_epoch.append(loss.item())

            # optimize
            optimizer.zero_grad()        
            loss.backward()

            # to prevent exploding gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        loss_curve.append(loss_epoch)

        # compute NDCG on validation set
        val_mean, _ = evaluate_ranknet(model, data.validation, device)
        ndcg_val_curve.append(val_mean)

        print(f"[Epoch {epoch}] loss: {loss.item()} validation ndcg: ({val_mean})")

        # early stopping using NDCG on validation set
        if not progress_over_last(ndcg_val_curve):
            break
        
    
    return model, loss_curve, ndcg_val_curve


def evaluate_ranknet(model, data_split, device, metric='ndcg'):

    scores = model.forward(torch.Tensor(data_split.feature_matrix).to(device))
    scores = scores.cpu().detach().numpy().reshape(-1)
    if metric:
        return evl.evaluate(data_split, scores)[metric]
    return evl.evaluate(data_split, scores)


if __name__ == "__main__":

    data = dataset.get_dataset().get_data_folds()[0]
    data.read_data()


    print('Number of features: %d' % data.num_features)
    print('Number of queries in training set: %d' % data.train.num_queries())
    print('Number of documents in training set: %d' % data.train.num_docs())
    print('Number of queries in validation set: %d' % data.validation.num_queries())
    print('Number of documents in validation set: %d' % data.validation.num_docs())
    print('Number of queries in test set: %d' % data.test.num_queries())
    print('Number of documents in test set: %d' % data.test.num_docs())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, loss_curve, ndcg_val_curve = train_ranknet(data, epochs=10, spedup=False, device = device)

    












