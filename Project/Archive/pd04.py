import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance as d

le = preprocessing.LabelEncoder()

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
item_raccomandabili = pd.read_csv('item_racc.csv', delimiter='\t').drop('Unnamed: 0',1)

users_r = pd.merge(user_profiles,target_users,on='user_id')
items_r = pd.merge(item_profiles,item_raccomandabili,on='id')

common_attributes = np.intersect1d(item_profiles.columns.values,user_profiles.columns.values).tolist()

users = users_r[common_attributes]
items = items_r[common_attributes]

users = users.fillna(0)
items = items.fillna(0)

users['country'] = users['country'].replace(0,'null')
users['country'] = le.fit_transform(users['country'])

items['country'] = items['country'].replace(0,'null')
items['country'] = le.fit_transform(items['country'])

items = items.astype(int)
users = users.astype(int)

def encode_feature(dataframe, to_encode):
    hot = preprocessing.OneHotEncoder(dtype='int')

    for feature in to_encode:
        a = hot.fit_transform(dataframe[feature].values.reshape((-1,1))).toarray()
        for i in range(np.shape(a)[1]):
            dataframe[feature + str(i)] = a[:,i]
    return

# encode_feature(items,common_attributes)
# encode_feature(users,common_attributes)
# items = items.drop(common_attributes,1)
# users = users.drop(common_attributes,1)
#
# for attribute in users.columns.difference(items.columns).tolist():
#     items[attribute] = pd.DataFrame(np.zeros((2486,)))

items = items.astype(int).sort_index(axis=1)
users = users.astype(int).sort_index(axis=1)

# cm = np.intersect1d(items.columns.values,users.columns.values).tolist()
# items = items[cm]
# users = users[cm]



def work():
    j=0
    suggestions = [0] * 10000
    for u in range(users.shape[0]):
        ratings = pd.DataFrame(columns=('user_id', 'item_id', 'rating'))
        user_id = users_r.ix[[u]]['user_id'].values[0]
        user = users.ix[[u]].values
        print(str(u))
        for i in range(items.shape[0]):
            item_id = items_r.ix[[i]]['id'].values[0]
            item = items.ix[[i]]
            rate = 1 - d.cosine(user,item)
            ratings.loc[j] = [user_id, item_id, rate]
            j = j + 1
        ratings[['user_id', 'item_id']] = ratings[['user_id', 'item_id']].astype(int)
        suggestions[u] = ratings[ratings['user_id'] == user_id].sort_values('rating',ascending=False).head(5)['item_id'].values



