from __future__ import division
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import preprocessing
import time
scaler = preprocessing.MaxAbsScaler()

items = pd.read_csv("item_profile.csv", delimiter='\t')
items_inactive = items[items['active_during_test'] == 0]
interactions = pd.read_csv('interactions.csv', delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
users = target_users['user_id'].values
user_profile = pd.read_csv('user_profile.csv', delimiter='\t')

ids = items['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size),uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

indici_target = [uds_to_index2.get(u) for u in users]

def interagiti(user_id):
    return np.unique(interactions[interactions['user_id'] == user_id]['item_id'].values)

anonimi = [u for u in users if interagiti(u).size<=0]
anonimi_indices = [uds_to_index.get(a) for a in anonimi]
attivi_indices = [uds_to_index.get(a) for a in users if interagiti(a).size>0]
users_attivi_indici = [uds_to_index.get(u) for u in users if interagiti(u).size>0]
users_anonimi_indici = [uds_to_index.get(u) for u in users if interagiti(u).size<=0]
users_attivi_indici2 = [uds_to_index2.get(u) for u in users if interagiti(u).size>0]

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def makeR(r,outputname):
    sugg = [0] * 10000
    print("start")
    for u in range(10000):
        R_u = r.getrow(u).toarray().ravel()
        R_u[list(items_inactive.index)] = 0 #rimuovi inattivi
        R_u[[ids_to_index.get(i) for i in interagiti(users[u])]] = 0 #rimuovi interagiti
        indices = R_u.argsort()[-5:][::-1]
        sugg[u] = [index_to_ids.get(i) for i in indices]
        if (u+1)%1000 == 0:
            print('Recommendations generated for: ' + str(u+1) + '/10000 users')

    result = np.column_stack((users,sugg))
    submission = open(outputname + '.csv', 'w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in result:
        for i in range(np.size(row)):
            if i == 0:
                line = str(row[0]) + ','
            elif i == 5:
                line = line + str(row[i])
            else:
                line = line + str(row[i]) + ' '
        submission.write(line + '\n')

    submission.close()
    return

def makeR2(R0,R1,outputname):
    sugg = [0] * 10000
    print("start")

    for u in attivi_indices:
        R_u = R0.getrow(u).toarray().ravel()
        R_u[list(items_inactive.index)] = 0 #rimuovi inattivi
        R_u[[ids_to_index.get(i) for i in interagiti(index_to_uds.get(u))]] = 0 #rimuovi interagiti
        indices = R_u.argsort()[-5:][::-1]
        sugg[u] = [index_to_ids.get(i) for i in indices]
        if (u+1)%1000 == 0:
            print('Recommendations generated for: ' + str(u+1) + '/10000 users')

    print('\n Filling anonimi. This may take a while...')
    for u in anonimi_indices:
        R_u = R1.getrow(u).toarray().ravel()
        R_u[list(items_inactive.index)] = 0 #rimuovi inattivi
        indices = R_u.argsort()[-5:][::-1]
        sugg[u] = [index_to_ids.get(i) for i in indices]

    result = np.column_stack((users,sugg))
    submission = open(outputname + '.csv', 'w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in result:
        for i in range(np.size(row)):
            if i == 0:
                line = str(row[0]) + ','
            elif i == 5:
                line = line + str(row[i])
            else:
                line = line + str(row[i]) + ' '
        submission.write(line + '\n')

    submission.close()
    print("done")
    return


def my_dot(l1,l2):
    result = sparse.csr_matrix((10000,167956))
    l = list(range(np.size(l1)))
    for i in l:
        for j in l:
            result += l1[i] * l2[i]
    return result

M = sparse.vstack([load_sparse_csr('CBF_tags_ALL1.npz'),load_sparse_csr('CBF_tags_ALL2.npz')])
time.sleep(5)
M *= 0.65
time.sleep(5)
M += 0.35 * load_sparse_csr('CBF_title_ALL.npz')
time.sleep(5)
M += 0.90 * load_sparse_csr('ALS_ALL.npz')[indici_target]
time.sleep(5)
M += 0.10 * load_sparse_csr('CF_ItemBased_GL_ALL.npz')[indici_target]
time.sleep(5)
M += 0.10 * load_sparse_csr('SVD4k_ALL.npz')[indici_target]
time.sleep(5)
M += 0.10 * load_sparse_csr('Collaborative_ALL.npz')
time.sleep(5)

N = 0.30 * load_sparse_csr('R_jobroles_norm_ALL.npz')
N += 0.10 * load_sparse_csr('R_TopPop.npz')[indici_target]
N += 0.20 * load_sparse_csr('R_edu_fieldofstudiesn_ALL.npz')
N += 0.10 *load_sparse_csr('RJR_ALL.npz')

makeR2(M,N,'submission')