import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import sklearn.metrics.pairwise as metrics
from sklearn import preprocessing
import math as m
from timestamp_processing import urm_time

# Return column indices of k highest values in sim_matrix
def top_k_items(sim_matrix, k):
    sim_matrix = sim_matrix.todense()
    indices = (-sim_matrix).argpartition(k, axis=None)[:k]
    x, y = np.unravel_index(indices, sim_matrix.shape)
    return y


# Load datasets
items = pd.read_table("item_profile.csv")
items = items[['id', 'active_during_test']]
interactions = pd.read_table("training_data.csv")

target_users_tot = pd.read_table("target_users.csv")

target_users = target_users_tot[:][target_users_tot['user_id'].isin(interactions['user_id'])]
target_users_no_int = target_users_tot[:][~target_users_tot['user_id'].isin(target_users['user_id'])]
target_users_no_int['recommended_items'] = 0


VALIDATION=False
# Uncomment these rows to perform validation
'''
VALIDATION = True
validation_users = pd.read_csv(val_path + "/validation_set2.csv")
interactions = pd.read_csv(val_path + "/new_interactions_2.csv")
target_users = validation_users
'''
# Add to the target dataset the column that will contain recommended items
target_users['recommended_items'] = 0

# Drop duplicates interactions and sort by user_id, item_id
interactions = interactions.drop_duplicates(['user_id', 'item_id'])

interactions['date'] = interactions['created_at']
ts = interactions['created_at']
min_ts = min(ts[ts != 0])
max_ts = max(ts)
ts = interactions['created_at']
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
ts_scaled = min_max_scaler.fit_transform(ts)
interactions['created_at'] = ts_scaled

interactions = interactions.sort_values(by=['user_id', 'item_id'])

# Get the list of unique user_ids that had at least one interactions
user_ids = interactions.drop_duplicates('user_id').reset_index()
# Assign to each user an incremental index
user_ids.loc[:, 'user_index'] = 0
user_ids['user_index'] = user_ids.index
user_ids = user_ids[['user_id', 'user_index']]
# Append that index to the interactions dataframe
interactions = pd.merge(interactions, user_ids, on='user_id')[['user_id', 'user_index', 'item_id','created_at']]
# Get the list of unique item_ids that were in at least one interaction
item_ids = interactions.drop_duplicates('item_id').reset_index()
# Assign to each item an incremental index
item_ids.loc[:, 'item_index'] = 0
item_ids['item_index'] = item_ids.index
item_ids = item_ids[['item_id', 'item_index']]
# Append that index to the interactions dataframe
interactions = pd.merge(interactions, item_ids, on='item_id')[['user_id', 'user_index', 'item_id', 'item_index','created_at']]
# Sort interactions by user_id and item_id
interactions = interactions.sort_values(by=['user_id', 'item_id'])
# Append the 'active_during_test' column from the item profile to the interactions dataframe
interactions['size'] = 0
interactions = pd.merge(interactions, items,
                        left_on='item_id', right_on='id',how='left')[['user_id', 'user_index', 'item_id',
                                                           'item_index', 'active_during_test','size','created_at']]

# Sort interactions by user_id and item_id
#interactions = interactions.sort_values(by=['user_id', 'item_id'])
# Extract the ids and indices of the active items
active_indices = interactions[interactions['active_during_test'] != 0][['item_id', 'item_index']].drop_duplicates('item_id')
active_indices = active_indices.fillna(0)

# Build URM by using as row indices the incremental indices that were associated to users
# and as column indices the indices that were associated to items
# in this way the user with index=0 will be the first row of the matrix indipendently from its user_id
# and the same is for items on columns
row_indices = interactions['user_index'].tolist()
col_indices = interactions['item_index'].tolist()
#values = [1]*len(col_indices)
values = interactions['created_at'].tolist()
URM = sp.coo_matrix((values, (row_indices, col_indices)),dtype=float)

#TODO: FIND USER_TARGET INDEX AND ACTIVE_ITEMS INDEX
target_indices = interactions[['user_id','user_index']][interactions['user_id'].isin(target_users['user_id'])].drop_duplicates().reset_index().drop('index',axis=1)

# Compute similarity as the dot product between the transposed URM and the URM itself
print("Computing similarity...")

u, s, vt = la.svds(urm_time, k = 300)
u = sp.csc_matrix(u)
vt = sp.csc_matrix(vt)
s_diag_matrix = sp.csc_matrix(np.diag(s))

#
# u = u[target_indices['user_index'].tolist(),:]
# vt = vt[:,active_indices['item_index'].tolist()]

x = s_diag_matrix.dot(vt)

R_out = sp.lil_matrix((10000,167956))

usersals = pd.read_csv("user_profile.csv", delimiter="\t")
itemsals = pd.read_csv("items_processed.csv", delimiter="\t")

users_dict = dict(zip(usersals['user_id'], usersals.index))
items_dict = dict(zip(itemsals['id'], itemsals.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict
items_nact_ids = itemsals[itemsals['active_during_test'] == 0]['id'].drop_duplicates().values
items_nact = [items_dict.get(i) for i in items_nact_ids]

l=10000
rusers = target_users['user_id'].values
index_to_uds = dict(zip(range(rusers.size),rusers))
uds_to_index = {v: k for k, v in index_to_uds.items()}

user_profile = pd.read_csv("user_profile.csv", delimiter='\t')
uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size),uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

for us in rusers:
    rec = u[uds_to_index2.get(us), :].dot(x)
    cols = np.argsort(rec)[::-1][:1000]
    R_out[uds_to_index.get(us)] = rec[cols]
    l -= 1
    print(l)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

save_sparse_csr('SVD4k',R_out.tocsr())
print("Done")

# final = pd.DataFrame(columns=['user_id','recommended_items'],dtype=int)
#
# intt = interactions.drop_duplicates('item_id').copy()
#
# for i in range(len(target_indices['user_index'])):
#     print(len(target_indices['user_id'])-i)
#     a = u[i,:]
#     sim_vect_i = np.array(np.dot(a,x).todense())[0]
#
#     rated = interactions[interactions['user_id'] == target_indices['user_id'][i]]
#     active_rated = list(rated[:][rated['active_during_test'] != 0]['item_index'])
#
#     for j in active_rated:
#         ar = int(active_indices['item_index'].tolist().index(j))
#         sim_vect_i[ar] = -20
#
#     top_k_index = np.argsort(sim_vect_i)[-30:][::-1]
#
#     top_k = pd.DataFrame({'indexx': top_k_index})
#
#     top_k['index_in_df'] = top_k['indexx'].apply(lambda x: active_indices['item_index'].iloc[x])
#
#     # Find item_id from item_index
#     top_k_ids = pd.merge(top_k, intt, how='left', right_on='item_index', left_on='index_in_df')[['item_id']]
#
#     final = final.append(pd.Series({'user_id':target_indices['user_id'][i],'recommended_items':' '.join(map(str,top_k_ids['item_id'].tolist()))}),ignore_index=True)
#
# final['user_id'] = final['user_id'].astype(int)