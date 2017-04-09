from project_utils import *
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sps
import random
import math

users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("items_processed.csv", delimiter="\t")
alpha = 0.5
rtp = 0.01

inter = interactions.drop('created_at', 1)
inter = inter[inter['interaction_type'] == 1]
inter = inter.drop_duplicates()

# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'], users.index))
items_dict = dict(zip(items['id'], items.index))
inv_map = {v: k for k, v in items_dict.items()}

# add users/items indexes columns
inter['user_index'] = [users_dict.get(i) for i in inter['user_id']]
inter['item_index'] = [items_dict.get(i) for i in inter['item_id']]


# items active_during_test=0 ids
nact_ids = items[items['active_during_test'] == 0]['id'].values
# items active_during_test=0 indexes
nact_index = [items_dict.get(k) for k in nact_ids]

urm = sps.csr_matrix((inter['interaction_type'], (inter['user_index'], inter['item_index'])))

sim = cosine_similarity(urm.transpose(), dense_output=False)
# sim = sim.tolil()
# sim.setdiag(0)
# sim = sim.tocsr()

rusers = target_users['user_id'].values

array_convergenza = np.zeros(1000)
array_convergenza[1] = 1 #mi serve per non farlo fermare al primo ciclo
numero_iterazioni = 0
convergiuto = False

def x(u,i):
    return np.dot(urm.getrow(u),sim.getcol(i)).toarray()[0][0]

X = np.dot(urm,sim)

racc = items['id'].values
DS = inter[['user_index','item_index']].values

# while not convergiuto:
#     u,i = random.choice(DS)
#     j = random.choice(racc)
#     j = items_dict.get(j)
#     xuij = x(u,i) - x(u,j)
#     e = math.exp(-xuij)
#     #old = sim[items_dict.get(i),items_dict.get(j)] * rtp
#     delta = alpha * (e/(1+e))
#     sim[i,j] -= delta
#     print(str(delta))

# weights the most clicked
weights = np.array(list(map(w,items['id'].values)),dtype='f')

sugg = [0] * 10000
for u in range(10000):
    us = rusers[u]
    uds = users_dict.get(us)
    ru = X.getrow(uds).toarray().ravel()
    nono_indices = [items_dict.get(i) for i in interagiti(u)]
    ru[nact_index] = 0
    ru[nono_indices] = 0
    ru *= weights
    indexes = ru.argsort()[-5:][::-1]
    sugg[u] = [inv_map.get(i) for i in indexes]
    print(str(u))

fillpop(sugg)
result = np.column_stack((rusers,sugg))
output(result)














