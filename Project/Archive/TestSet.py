import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import Radond_sugg

K = 50
k = 5

userss = pd.read_csv("target_users.csv", delimiter="\t")
users = [val for sublist in userss.values for val in sublist]
interactions_raw = pd.read_csv("interactions.csv", usecols=['user_id', 'item_id','interaction_type', 'created_at'], delimiter="\t")
items_raw = pd.read_csv("item_profile.csv", delimiter="\t")
items = pd.read_csv("item_profile.csv", usecols=['id','career_level', 'discipline_id', 'latitude', 'longitude','industry_id', 'employment', 'active_during_test'], delimiter="\t")
items = items[items['active_during_test'] == 1]
items = items[items.columns.difference(['active_during_test'])].reset_index()
items = items.fillna(0) #fills missing data with zeros

interactions = interactions_raw[interactions_raw['item_id'].isin(items['id'])].reset_index()
interactions = interactions[interactions.columns.difference(['index'])]
test_set = pd.DataFrame(columns=('user_id', 'item_id', 'interaction_type', 'created_at'))
interactions = interactions.sort_values('user_id')

for user in users:
    test_set = test_set.append(interactions[interactions['user_id'] == user].sort_values('created_at',False).head(5))

test_set = test_set.astype(int)

test_set.to_csv('test_set.csv', sep='\t')
