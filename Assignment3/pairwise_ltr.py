import dataset
import torch
from torch import nn
import time
import numpy as np
from itertools import permutations 
import matplotlib.pyplot as plt
import evaluate as evl

from pointwise_ltr import progress_over_last
from lambdarank import create_matrices


class RankNet(nn.Module):
    def __init__(self, n_hidden, gamma, batch_size, device):  

        super().__init__()

        self.gamma = gamma
        self.batch_size = batch_size

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


    def loss(self, scores, labels):

        _, S = create_matrices(scores, self.gamma, labels)
        score_diff = scores - scores.t()

        # Assignment Equation (3)
        C = 0.5 * (1 - S) * self.gamma * score_diff + torch.log2(1 + torch.exp(-self.gamma * score_diff))

    
        # # WITH SAMPLING FIXED BATCH SIZE
        # B = np.ones(C.shape) - np.eye(C.shape[0])
        # idx = B.nonzero()
        # sample = [np.random.randint(0, len(idx[0])) for i in range(self.batch_size)]

        # # i = np.arange(len(idx[0]))
        # # np.random.shuffle(i)
        # # sample = i[:self.batch_size]
        # idx1, idx2 = idx[0][sample], idx[1][sample]
    
        # C_T = C[idx1, idx2].sum()

        # return C_T

        #WITH MEAN 
        # pairs on the diagonal are not valid
        C_T = torch.sum(C  * (torch.ones_like(C) - torch.eye(C.shape[0]))) 
        C_mean = C_T / (C.nelement() - C.shape[0])

        return C_mean

    def spedup_loss(self, scores, labels, ):
        Sc, S = create_matrices(scores, self.gamma, labels)
        _lambda = self.gamma * (0.5 * (1 - S) - Sc)

        # set diagonal of lambda to zero 
        lambda_0 = _lambda  * (torch.ones_like(_lambda, dtype=torch.float) - torch.eye(_lambda.shape[0], dtype=torch.float)) 

        # average across i 
        lambdas = (lambda_0.sum(dim=1) / (lambda_0.shape[1] - 1))[:, None]

        return (scores * lambdas.detach()).mean()

            

def get_samples_by_qid(qid, data_split, device):

    docs = data_split.query_feat(qid)
    labels = data_split.query_labels(qid)

    return torch.Tensor(docs).to(device), torch.Tensor(labels).to(device) 


def train_ranknet(data, epochs=100, n_hidden=1024, gamma=1.0, lr=1e-4, batch_size=50, spedup = True, device='cpu'):


    model = RankNet(n_hidden=n_hidden, gamma=gamma, batch_size=batch_size, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    if spedup:
        loss_fn = model.spedup_loss
    else:
        loss_fn = model.loss

    # track loss and ndcg on validation set 
    loss_curve = []
    ndcg_val_curve = []

    queries = np.arange(0, data.train.num_queries())

    
    for epoch in range(epochs): 

        loss_epoch = []
        np.random.shuffle(queries)
     
        for qid in queries[:100]: 

            docs, labels = get_samples_by_qid(qid=qid, data_split=data.train, device=device)

            # if there is only one doc retrieved by the query, sample another query
            if docs.shape[0] < 2: 
                other_q = queries[100::]
                np.random.shuffle(other_q)
                for qid_new in other_q:
                    docs, labels = get_samples_by_qid(qid=qid_new, data_split=data.train, device=device)
                    if not docs.shape[0] < 2: 
                        break

            scores = model.forward(docs)

            loss = loss_fn(scores, labels)

            if spedup:
                loss_report = model.loss(scores, labels).item()
                loss_epoch.append(loss_report)
            else:
                loss_report = loss.item()
                loss_epoch.append(loss_report)

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

        print(f"[Epoch {epoch}] loss: {loss_report} validation ndcg: ({val_mean})")

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

def plot_pairwise_ltr(model, loss_curve, ndcg_val_curve):
    x = np.linspace(0, len(ndcg_val_curve) + 1, len(ndcg_val_curve))

    loss_means = np.array([np.mean(l) for l in loss_curve])
    loss_stds = np.array([np.std(l) for l in loss_curve])

    # plt.title("Loss vs. NDCG on Validation Data")
    # plt.xlabel("Epochs")
    # plt.plot(x, loss_means, label='loss over epochs')
    # plt.fill_between(x, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2)
    # plt.plot(x, ndcg_val_curve, label='NDCG on validation data')
    # plt.legend()

    plt.title("NDCG on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, ndcg_val_curve, label='NDCG on validation data')
    plt.legend()
    plt.savefig('pairwise_NDCG')

    plt.clf()
    plt.title("Loss on Validation Data")
    plt.xlabel("Epochs")
    plt.plot(x, loss_means, label='loss over epochs')
    plt.fill_between(x, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2)
    plt.legend()
    plt.savefig('pairwise_loss')



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


    # tic = time.perf_counter()
    model, loss_curve, ndcg_val_curve = train_ranknet(data, epochs=100, spedup=False, device = device,  lr=1e-5)

    # toc = time.perf_counter() 
    # print(f"Trained in {toc - tic:0.4f} seconds")


    test_mean = evaluate_ranknet(model, data.test, device, metric=None)

    print(test_mean)

    plot_pairwise_ltr(model, loss_curve, ndcg_val_curve)

    












