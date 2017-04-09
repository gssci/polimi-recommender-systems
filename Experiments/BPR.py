from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import nltk
from scipy import sparse
import random

target_users = pd.read_csv("target_users.csv", delimiter='\t')
interactions = pd.read_csv('training_data.csv', delimiter='\t')
user_profile = pd.read_csv('user_profile.csv', delimiter='\t')
item_profile = pd.read_csv('item_profile.csv', delimiter='\t')

users = target_users['user_id'].values
ids = item_profile['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size),uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


X = load_sparse_csr('ALS4k.npz')
X = X.tolil()

users_dict = dict(zip(user_profile['user_id'], user_profile.index))
items_dict = dict(zip(item_profile['id'], item_profile.index))

inter = interactions.copy()
inter['interaction_type'] = 1
inter = inter.drop('Unnamed: 0',1)
inter = inter.drop('created_at',1)
inter = inter.drop_duplicates()
inter['user_index'] = [users_dict.get(i) for i in inter['user_id']]
inter['item_index'] = [items_dict.get(i) for i in inter['item_id']]
urm = sparse.csr_matrix((inter['interaction_type'], (inter['user_index'], inter['item_index'])))

def user_uniform_item_uniform_sampling(R, size, replace=True, seed=1234):
    col_indices = R.indices
    indptr = R.indptr
    M = R.shape[0]
    N = R.shape[1]
    nnz = len(R.data)

    sample = np.zeros((size, 3), dtype=np.int64)
    if not replace:
        is_sampled = np.zeros(nnz, dtype=np.int8)

    np.random.seed(seed)

    i = 0
    while i < size:
        # 1) sample a user from a uniform distribution
        iid = np.random.choice(M)

        # 2) sample a positive item uniformly at random
        start = indptr[iid]
        end = indptr[iid + 1]
        pos_candidates = col_indices[start:end]
        if start == end:
            # empty candidate set
            continue
        if replace:
            # sample positive items with replacement
            jid = np.random.choice(pos_candidates)
        else:
            # sample positive items without replacement
            # use a index vector between start and end
            aux = np.arange(start, end)
            if np.all(is_sampled[aux]):
                # all positive items have been already sampled
                continue
            idx = np.random.choice(aux)
            while is_sampled[idx]:
                # TODO: remove idx from aux to speed up the sampling
                idx = np.random.choice(aux)
            is_sampled[idx] = 1
            jid = col_indices[idx]

        # 3) sample a negative item uniformly at random
        # build the candidate set of negative items
        # TODO: precompute the negative candidate set for speed-up
        neg_candidates = np.delete(np.arange(N), pos_candidates)
        kid = np.random.choice(neg_candidates)
        sample[i, :] = [iid, jid, kid]
        i += 1
        if i % 10000 == 0:
            print('Sampling... {:.2f}% complete'.format(i / size * 100))

    return sample
#
# def transform_to_id(x):
#     return [index_to_uds2.get(x[0]),index_to_ids.get(x[1]),index_to_ids.get(x[2])]
#
# def xuij(u,i,j):
#     return (X[u,i] - X[u,j])

def bpr():
    size = len(urm.data)
    sample = user_uniform_item_uniform_sampling(urm, size, True, np.random.randint(100,high=1000))
    loss = 0.0
    lrate = 0.01
    lrate_decay = 1.0
    user_reg = 0.9
    pos_reg = 0.0001
    neg_reg = 0.4
    iters = 10
    for it in range(iters):
        print("Iteration: " + str(it + 1))
        for n in range(size):
            u, i, j = sample[n]
            # get the user and item factors
            X_u = X.getrow(u).toarray().ravel().copy()
            X_ui = X_u[i].copy()
            X_uj = X_u[j].copy()

            # compute the difference of the predicted scores
            xuij = X_ui - X_uj
            # compute the sigmoid
            sig = 1. / (1. + np.exp(-xuij))
            # update the loss
            loss += np.log(sig)

            deriv = 1. - sig
            X[u,i] += lrate * (deriv * xuij - pos_reg * X_ui)
            X[u,j] += lrate * (deriv * xuij - neg_reg * X_uj)

        loss /= size
        lrate *= lrate_decay
        sample = user_uniform_item_uniform_sampling(urm, size, True, np.random.randint(100,high=1000))


