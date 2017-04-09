from __future__ import division

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import preprocessing
from scipy.optimize import minimize
import map5

abs = preprocessing.MaxAbsScaler()

item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
items_inactive = item_profiles[item_profiles['active_during_test'] == 0]
interactions = pd.read_csv("training_data.csv", delimiter='\t')
indici_inattivi = list(items_inactive.index)
target_users = pd.read_csv("target_users.csv", delimiter='\t')
items_active = item_profiles[item_profiles['active_during_test'] == 1]
users = target_users['user_id'].values
user_profile = pd.read_csv('user_profile.csv', delimiter='\t')

ids = item_profiles['id'].values
index_to_ids = dict(zip(range(ids.size), ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size), users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size), uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

indici_target = [uds_to_index2.get(u) for u in users]


def interagiti(user_id):
    return np.unique(interactions[interactions['user_id'] == user_id]['item_id'].values)


anonimis = [u for u in users if interagiti(u).size <= 0]
users_attivi_indici = [uds_to_index.get(u) for u in users if interagiti(u).size > 0]
users_anonimi_indici = [uds_to_index.get(u) for u in anonimis]
users_attivi_indici2 = [uds_to_index2.get(u) for u in users if interagiti(u).size > 0]
users_indici2 = [uds_to_index2.get(u) for u in users]


def items_id_raccomandabili(user_id):
    raccomandabili = np.unique(items_active[~items_active.id.isin(interagiti(user_id))]['id'].values)
    return raccomandabili


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


indici_interagiti = [0] * 10000

for u in range(10000):
    indici_interagiti[u] = [ids_to_index.get(i) for i in interagiti(users[u])]



def fillanon(sugg):
    # dati = pd.read_csv('outa5noano.csv', delimiter='\,', engine='python')
    # dati = dati['recommended_items'].values
    most_popular = [2778525, 1244196, 1386412, 657183, 2791339]

    for i in users_anonimi_indici:
        sugg[i] = most_popular #dati[i].split(' ')  #
    return


def makeR(r, outputname):
    sugg = [0] * 10000

    for u in users_attivi_indici:
        R_u = r.getrow(u).toarray().ravel()
        R_u[indici_inattivi] = 0  # rimuovi inattivi
        R_u[indici_interagiti[u]] = 0  # rimuovi interagiti
        indices = R_u.argsort()[-5:][::-1]
        sugg[u] = [index_to_ids.get(i) for i in indices]

    fillanon(sugg)
    result = np.column_stack((users, sugg))
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

# TopPop = abs.fit_transform(TopPop.T).T

def makeRext(r, ran, outputname, allFromRan=False):
    sugg = [0] * 10000

    if allFromRan:
        for u_id in users:
            u = uds_to_index.get(u_id)
            R_u = ran.getrow(u).toarray().ravel()
            if np.size(R_u.nonzero()) > 0:
                R_u[indici_inattivi] = 0  # rimuovi inattivi
                R_u[indici_interagiti[u]] = 0  # rimuovi interagiti
                indices = R_u.argsort()[-5:][::-1]
            else:
                indices = range(5)
            sugg[u] = [index_to_ids.get(i) for i in indices]
    else:
        for u in users_attivi_indici:
            # R_u = r.getrow(u).toarray().ravel()
            # if np.size(R_u.nonzero()) > 0:
            #     R_u[indici_inattivi] = 0  # rimuovi inattivi
            #     R_u[indici_interagiti[u]] = 0  # rimuovi interagiti
            #     indices = R_u.argsort()[-5:][::-1]
            # else:
            #     indices = range(5)
            sugg[u] = [-1,-1,-1,-1,-1]
            # if (u+1)%1000 == 0:
            #    print('Recommendations generated for: ' + str(u+1) + '/10000 users')

        for u in users_anonimi_indici:
            R_a = ran.getrow(u).toarray().ravel()
            R_a[indici_inattivi] = 0
            indices = R_a.argsort()[-5:][::-1]
            sugg[u] = [index_to_ids.get(i) for i in indices]

    result = np.column_stack((users, sugg))
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


def makeEval(matrice):
    makeR(matrice, 'OutR0')
    val = map5.evaluate_submission('OutR0.csv')
    print(str(val))
    return val


def makeevalext(matrice):
    makeRext(R0, matrice, 'outano')
    val = map5.evaluate_submission('outano.csv')
    print(str(val))
    return val


R0 = sparse.csr_matrix((10000, 167956))
TopPop = load_sparse_csr('R_TopPopLight.npz')[users_indici2]

ALS = load_sparse_csr('ALS.npz')[users_indici2]
CF = load_sparse_csr('CF_ItemBased_GL.npz')[users_indici2]  # User-based CF * TopPop weighted
SVD = load_sparse_csr('SVD.npz')[users_indici2]
# RJR = load_sparse_csr('R_jobroles_norm.npz')
# RJR0 = load_sparse_csr('RJR.npz')
# RFS = load_sparse_csr('R_edu_fieldofstudiesn.npz')
# RFS0 = load_sparse_csr('RFS.npz')
CFP = load_sparse_csr('Collaborative.npz')
# CBF_tags = sparse.vstack([load_sparse_csr('CBF_tags1_noT.npz'),load_sparse_csr('CBF_tags2_noT.npz')])
# CBF_title = load_sparse_csr('CBF_title_noT.npz')

def f(w):
    return -makeEval(w[0]*ALS + w[1]*CF + w[2]*SVD + (w[3]*CBF_tags + w[4]*CBF_title) + (w[5]*RJR + w[6]*RFS))

# def g(w):
#     return -makeevalext(w[0]*RJR + w[1]*RFS + w[2]*TopPop +w[3]*RJR0 + w[4]*RFS0)

#makeEval(0.9 * ALS + 0.3 * RX + 0.65 * RTA + 0.35 * RTI + 0.1 * RJR)

# res = minimize(g, np.array([0, 0, 0,0,0]),method='Nelder-Mead', tol=0.0001, options={'disp': True})
#res = minimize(f, np.array([0,0,0,0,0,0,0]),method='Nelder-Mead', tol=0.0001, options={'disp': True})


# CF_ItemBased = load_sparse_csr('CF_ItemBased_GL.npz')[users_indici2] #0.004057
# CF_ItemBased3 = load_sparse_csr('CF_ItemBased_UT.npz')[users_indici2]
# CF_ItemBased4 = load_sparse_csr('CF_URMTIME_ItemBased.npz')[users_indici2]
# CF_ItemBased2 = load_sparse_csr('CF_II.npz')[users_indici2] #0.00397933333333
#
# makeEval(CF_ItemBased)
# makeEval(CF_ItemBased2)
#
# makeEval(0.9*ALS + 0.3*CF_ItemBased + 0.65*RTA + 0.35*RTI + 0.1*RJR)