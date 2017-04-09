from __future__ import division
import numpy as np
import pandas as pd
from scipy import sparse
from sqlalchemy.orm.strategy_options import loader_option

import map5
import math
from sklearn import preprocessing
abs = preprocessing.MaxAbsScaler()

from scipy.optimize import minimize

item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
items_inactive = item_profiles[item_profiles['active_during_test'] == 0]
interactions = pd.read_csv("training_data.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
items_active = item_profiles[item_profiles['active_during_test'] == 1]
users = target_users['user_id'].values
user_profile = pd.read_csv('user_profile.csv', delimiter='\t')

ids = item_profiles['id'].values
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

anonimis = [u for u in users if interagiti(u).size<=0]
users_attivi_indici = [uds_to_index.get(u) for u in users if interagiti(u).size>0]
users_anonimi_indici = [uds_to_index.get(u) for u in anonimis]
users_attivi_indici2 = [uds_to_index2.get(u) for u in users if interagiti(u).size>0]
users_indici2 = [uds_to_index2.get(u) for u in users]

def items_id_raccomandabili(user_id):
    raccomandabili = np.unique(items_active[~items_active.id.isin(interagiti(user_id))]['id'].values)
    return raccomandabili

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

indici_interagiti = [0] * 10000

for u in range(10000):
    indici_interagiti[u] = [ids_to_index.get(i) for i in interagiti(users[u])]

indici_inattivi = list(items_inactive.index)

def fillanon(sugg):
    dati = pd.read_csv('outano2.csv', delimiter='\,', engine='python')
    dati = dati['recommended_items'].values

    for i in users_anonimi_indici:
        sugg[i] = dati[i].split(' ')
    return

def makeR(r,outputname):
    sugg = [0] * 10000

    for u in users_attivi_indici:
        R_u = r.getrow(u).toarray().ravel()
        R_u[indici_inattivi] = 0 #rimuovi inattivi
        R_u[indici_interagiti[u]] = 0 #rimuovi interagiti
        indices = R_u.argsort()[-5:][::-1]
        sugg[u] = [index_to_ids.get(i) for i in indices]

    fillanon(sugg)
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

def makeRext(r,ran,outputname):
    sugg = [0] * 10000

    for u in users_attivi_indici:
        R_u = r.getrow(u).toarray().ravel()
        if np.size(R_u.nonzero()) > 0:
            R_u[indici_inattivi] = 0 #rimuovi inattivi
            R_u[indici_interagiti[u]] = 0 #rimuovi interagiti
            indices = R_u.argsort()[-5:][::-1]
        else:
            indices = range(5)
        sugg[u] = [index_to_ids.get(i) for i in indices]
        #if (u+1)%1000 == 0:
        #    print('Recommendations generated for: ' + str(u+1) + '/10000 users')

    for u in users_anonimi_indici:
        R_a = ran.getrow(u).toarray().ravel()
        R_a[indici_inattivi] = 0
        indices = R_a.argsort()[-5:][::-1]
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
    return

def makeEval(matrice):
    makeR(matrice,'OutR')
    val = map5.evaluate_submission('OutR.csv')
    print(str(val))
    return val

def makeEvalExt(matrice):
    makeRext(R0,matrice,'OutRExt')
    val = map5.evaluate_submission('OutRExt.csv')
    print(str(val))
    return val


R0 = sparse.csr_matrix((10000,167956))

# RTA = load_sparse_csr('RTA4k.npz')[users_indici2]
# RTA = abs.fit_transform(RTA.T).T
# RTI = load_sparse_csr('RTI4k.npz')[users_indici2]
# RTI = abs.fit_transform(RTI.T).T
# ALS = load_sparse_csr('ALS4k.npz')[users_indici2]
# ALS = abs.fit_transform(ALS.T).T
# GL = load_sparse_csr('R_GL4k.npz')[users_indici2]
# GL = abs.fit_transform(GL.T).T

def normalizza(matrix):
    for i in range(matrix.shape[0]):
        max = matrix[i].max()
        if max>0:
            matrix[i] = matrix[i] / max

# ALS = sparse.lil_matrix((10000,167956))
# for (x,y) in zip(users_attivi_indici,users_attivi_indici2):
#     ALS[x] = R_ALS[y]
#     print (str((x,y)))
#
# R7 = load_sparse_csr('R74k.npz')[users_indici2]
# R7 = abs.fit_transform(R7.T).T
#
# ##ANONIMI
# TopPop = load_sparse_csr('R_TopPopLight.npz')[users_indici2]
# TopPop = abs.fit_transform(TopPop.T).T
# RJR = load_sparse_csr('R_jobroles.npz')
# RJ0 = load_sparse_csr('R_edu_fieldofstudies.npz')
#
# #makeEval(0.90*ALS + 0.50*RX + 0.10*RUS + 0.20*R7 + 0.50*RTA + 0.15*RTI) # MAP@5: 0.0101566666667
# RRR = [ALS,RTA,RTI]

def my_dot(l1,l2):
    result = sparse.csr_matrix((10000,167956))
    l = list(range(np.size(l1)))
    for i in l:
        for j in l:
            result = result + (l1[i] * l2[i])
    return result

def f(w):
    return -makeEval(w[0]*ALS+w[1]*R7+w[2]*RTA+w[3]*RTI)

def fext2(w):
    makeRext(R0,my_dot(w,[RJR,RJ0,TopPop]),'outano2')
    val = map5.evaluate_submission('outano2.csv')
    print(str(val))
    return -val

w5 = [3.33280797e-04,   8.86411447e-05,   1.48361095e-04, 2.92006172e-05]
w_max = [4.05432208e-04,   1.05941144e-04,   1.62917407e-04,
         4.56303119e-05,   1.32362393e-05]
res = minimize(f,w5, method='Nelder-Mead', options={'disp': True})
#resext2 = minimize(fext2,wa, method='Nelder-Mead', options={'disp': True})

#RRR = [ALS,R7,RTA,RTI,TopPop]
 #Current function value: -0.01127 - ALS senza normalizzazione
# w5 = [3.33280797e-04,   8.86411447e-05,   1.48361095e-04,
#          2.92006172e-05]
#
# def save_sparse_csr(filename,array):
#     np.savez(filename,data = array.data ,indices=array.indices,
#              indptr =array.indptr, shape=array.shape )

#wx = res.x
#print(wx)

#
# wfuck = [2.06459963e-04,   6.26909627e-05,   6.70459979e-05,
#          1.98775755e-05,   5.32386503e-06,   5.30168333e-06]
#
# WANON = [0.00026575,0.00027226,0.00012546]


wfuck = [2.06459963e-04,   6.26909627e-05,   6.70459979e-05,
         1.98775755e-05,   5.32386503e-06,   5.30168333e-06]

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
            indptr =array.indptr, shape=array.shape )
