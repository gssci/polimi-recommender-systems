import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

interactions = pd.read_csv('interactions.csv', delimiter="\t")
target_users = pd.read_csv("target_users.csv", delimiter='\t')
it = pd.read_csv('item_profile.csv', delimiter="\t")['active_during_test'].values
item_profiles = pd.read_csv("items_preprocessed_ASFUCK.csv", delimiter=',').drop('Unnamed: 0',1)

items = sparse.csc_matrix(item_profiles.drop('id',1))

ids = item_profiles['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}

def user_profile(user_id):
    interazioni = interactions[interactions['user_id'] == user_id]
    jobs = interazioni['item_id'].values
    pesi = interazioni['interaction_type'].values
    l = interazioni['item_id'].values.size
    if l>0:
        profile = [0] * 425
        profile = sparse.csr_matrix(profile)
        for i in range(l):
            profile = profile + items.getrow(ids_to_index.get(jobs[i])) * pesi[i]
        profile = profile / l
        similarity = profile.dot(items.T)
        similarity = np.asarray(np.multiply(similarity.todense(),it))[0]
        indices = similarity.argsort()[-5:][::-1]
        sugg = [index_to_ids.get(i) for i in indices]
    else:
        sugg = [0,0,0,0,0]
    return sugg

def recommend():
    suggestions = [0] * 10000
    userss = target_users['user_id'].values
    for i in range(10000):
        suggestions[i] = user_profile(userss[i])
        print(str(i))

    result = np.column_stack((userss, suggestions))
    return result


