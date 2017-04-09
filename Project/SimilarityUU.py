import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
from scipy.spatial import distance as d
import project_utils as util
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# #utility di sklearn che codifica attributi in forma di stringa, la usiamo per convertire le nazioni in numeri in modo coerente e confrontabile
# le = preprocessing.LabelEncoder()
#
# user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
# target_users = pd.read_csv("target_users.csv", delimiter='\t')
#
# #set da cui dopo recupero gli user_id e l'id degli items, ristretti agli items attivi e ai target users
# users_r = pd.merge(user_profiles,target_users,on='user_id')
#
# #rimuovo i valori nulli e li sostituisco con 0
# users = users_r.fillna(0)
#
# users['country'] = users['country'].replace(0,'null')
# users['country'] = le.fit_transform(users['country'])
#
# users = util.split_string_column(users,'edu_fieldofstudies',True)
# users = util.split_string_column(users,'jobroles',True)
#
# users = users.drop('edu_fieldofstudies',1)
# users = users.drop('jobroles',1)
# users = users.drop('0',1)
#
# to_encode = ['career_level','discipline_id','industry_id','country','region','edu_degree']
# util.encode_feature(users,to_encode)
#
# users = users.drop(to_encode,1)
# users = users.drop(['experience_n_entries_class','experience_years_experience','experience_years_in_current'],1)

users = pd.read_csv("users_preprocessed.csv", delimiter=',').drop('Unnamed: 0',1)
user_matrix = sparse.csc_matrix(users.drop('user_id',1))

similarity = cosine_similarity(user_matrix,dense_output=False)

ids = users['user_id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
users_con_interactions = pd.unique(pd.merge(pd.read_csv('target_users.csv',delimiter='\t'),pd.read_csv('interactions.csv',delimiter='\t'))['user_id'])
indici_users_con_interactions = [ids_to_index.get(u) for u in users_con_interactions]

#similarity[:,indici_users_con_interactions] = 0

def get_K_similar_users(K,user_id):
    ind = ids_to_index.get(user_id)
    sims = similarity.getrow(ind).toarray()[0]
    sims[ind] = 0
    sims[indici_users_con_interactions] = 0
    indices = sims.argsort()[-K:][::-1]
    result = [index_to_ids.get(i) for i in indices]
    return result

