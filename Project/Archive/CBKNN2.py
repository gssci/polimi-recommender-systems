import Radond_sugg

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

K = 25
le = preprocessing.LabelEncoder()

##IMPORT AND PREPROCESS##
users = pd.read_csv("target_users.csv", delimiter="\t")
target = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)

user_profiles = pd.read_csv('user_profile_ppd.csv', delimiter="\t")
user_profiles = user_profiles.drop('Unnamed: 0',1)

item_profiles = pd.read_csv("item_profile.csv", delimiter="\t")
item_profiles = item_profiles.fillna(0)
item_profiles['country'] = item_profiles['country'].replace(0,'zero')
item_profiles['country'] = le.fit_transform(item_profiles['country'])

def encode_feature(dataframe, to_encode):
    hot = preprocessing.OneHotEncoder()

    for feature in to_encode:
        a = hot.fit_transform(dataframe[feature].values.reshape((-1,1))).toarray()
        for i in range(np.shape(a)[1]):
            dataframe[feature + str(i)] = a[:,i]
    return

to_encode = ['discipline_id','industry_id','country','region','career_level','employment']
encode_feature(item_profiles,to_encode)
item_profiles = item_profiles.drop(['title','tags','discipline_id','industry_id','latitude','longitude','created_at','country','region','career_level','employment'],1).astype(int)


interactions = pd.read_csv('interactions.csv', delimiter="\t")
item_raccomandabili = item_profiles[item_profiles['active_during_test'] == 1].drop('active_during_test',1).reset_index().drop('index',1)
interazioni_item = interactions['item_id'].drop_duplicates().reset_index().drop('index',1)
# Grab DataFrame rows where column doesn't have certain values
value_list = interazioni_item['item_id'].values
item_raccomandabili = item_raccomandabili[~item_raccomandabili.id.isin(value_list)]


feed = pd.merge(interactions,item_profiles.rename(columns = {'id':'item_id'}),on='item_id').drop('user_id',1)

# interactions = pd.merge(interactions,itemsKKK.rename(columns = {'id':'item_id'}),on='item_id')
# top = interactions.groupby(['user_id', 'item_id']).aggregate(np.sum).drop('created_at',1).reset_index().sort(['user_id','interaction_type'],ascending=[1,0])
# top = pd.merge(top,users,on='user_id')
############################


########RATING ESTIMATIONS FOR THE GODS OF COMPUTING############
ratings = pd.DataFrame(columns=('user_id', 'item_id', 'rating'))

#training set := items for which we have an entry in interactions
training = pd.merge(interactions,item_profiles,left_on='item_id',right_on='id')
training = training.drop('created_at',1)
training = training.drop('item_id',1)
training = training.groupby(training.columns.difference(['user_id','interaction_type']).tolist()).aggregate(np.max).reset_index() #cazzo sono il dio di pandas

known_ratings = training[['id','user_id','interaction_type']].copy()

training2 = item_profiles[item_profiles.columns.difference(['id','interaction_type','active_during_test'])]
training3 = pd.merge(known_ratings,item_profiles, on='id').drop(['user_id','interaction_type','active_during_test'], 1)
tmp = training3.drop('id', 1)
items_nn = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(tmp)
item_profiles = item_profiles[item_profiles.columns.difference(['active_during_test'])] #DA RIVEDERE
#######################


def r(u,j):
    r = 0
    try:
       r = known_ratings[(known_ratings['id'] == j) & (known_ratings['user_id'] == u)]['interaction_type'].values[0]
    except IndexError:
       print('missing u: ' + str(u) + ' j: ' + str(j))
    return r

def itembyID(id):
    return item_profiles[item_profiles['id'] == id]



def workworkworkworkwork():

    def estimate_rating(user, item):
        x = itembyID(item)
        x = x[x.columns.difference(['id'])]
        distances, indices = items_nn.kneighbors(x)
        similarities = 1 - distances
        vicini = training3.ix[indices[0]]['id'].values

        num = 0
        den = 0

        for j in range(vicini.size):
            num = num + (r(user, vicini[j]) * similarities[0][j])
            den = den + similarities[0][j]
        return num / den

    for i in range(2):
        user = Radond_sugg.choice(target)
        j = 0
        for item in item_raccomandabili['id'].values:
            r_est = estimate_rating(user,item)
            ratings.loc[j] = [user, item, r_est]
            j = j + 1
            print('ehi')
        print('####################################remaining: ' + str(i-1))

    return




