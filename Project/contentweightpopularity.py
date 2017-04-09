import pandas as pd
import numpy as np
import contentweight

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')

userss = target_users['user_id'].values
interactions = pd.read_csv('interactions.csv', delimiter="\t")
a = np.unique(interactions['user_id'].values)
anonimi = [u for u in userss if u not in a]

def weight(i):
    return 1 - 1/i**(1/2)

interactionz = interactions.groupby('item_id').aggregate(np.sum).reset_index().sort_values('interaction_type', ascending=False).drop('user_id',1)
ws = dict(zip(interactionz['item_id'].values,np.apply_along_axis(weight,0,interactionz['interaction_type'].values)))

def w(i):
    x = 0.01
    try:
        x = ws.get(i)
    except KeyError:
        x = 0.01
    return x

item = pd.read_csv('item_profile.csv', delimiter='\t')
item_attivi = item[item['active_during_test'] == 1]['id'].values

interactions = interactions[interactions['interaction_type'] == 1]
interactions = interactions[interactions.item_id.isin(item_attivi)]

top = interactions.drop('user_id',1).groupby('item_id').aggregate(np.sum).sort_values('interaction_type',ascending=False).head(70)
top = top.reset_index().drop('created_at',1)

keys = top['item_id'].values
weights = [w(i) for i in keys]
top['weights'] = weights

def recommend(user_id):
    topu = top.copy()
    wpu = [contentweight.probability(user_id, item_id) for item_id in keys]
    topu['weights'] = topu['weights'] * wpu
    return topu.sort_values('weights',ascending=False).head(5)['item_id'].values

def fillcwp(sugg):
    uds = target_users['user_id'].values
    uds_to_index = dict(zip(uds, range(uds.size)))
    for u in anonimi:
        i = uds_to_index.get(u)
        sugg[i] = recommend(u)
    return
