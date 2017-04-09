import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import Radond_sugg

K = 90
le = preprocessing.LabelEncoder()

users = pd.read_csv("target_users.csv", delimiter="\t")

user_profiles = pd.read_csv('user_profile_ppd.csv', delimiter="\t")
user_profiles = user_profiles.drop('Unnamed: 0',1)
item_profiles = pd.read_csv("item_profile.csv", delimiter="\t")
item_profiles = item_profiles.fillna(0)

item_profiles['country'] = item_profiles['country'].replace(0,'zero')
item_profiles['country'] = le.fit_transform(item_profiles['country'])



#TRAIN OVER ITEMS INTERSEZIONE INTERACTIONS INCLUDING THE INACTIVE ONES
#THEN, WE NEED TO COMPUTE THE COLUMN USER X ITEM X RATING


tmp = user_profiles[user_profiles.columns.difference(['user_id'])]
tmp2 = item_profiles[item_profiles.columns.difference(['id'])]

users_nn = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(tmp)
#items_nn = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(tmp2)


def closestusers(x): #nb prende in input una riga del database
    x = x[x.columns.difference(['user_id'])]
    distances, indices = users_nn.kneighbors(x)
    return user_profiles.ix[indices[0]]

#RETURN ITEMS FOR WHICH WE HAVE A INTERACTION TYPE PORCODIO
def closestitems(x):
    x = x[x.columns.difference(['id'])]
    distances, indices = items_nn.kneighbors(x)
    return item_profiles.ix[indices[0]]

def userbyID(id):
    return user_profiles[user_profiles['user_id'] == id]

def itembyID(id):
    return item_profiles[item_profiles['id'] == id]


suggestions = [0] * 10000
j = 0
#
# for i in range(10000):
#     user = users['user_id'].values[i]
#
#     try:
#         most_i = top[top['user_id'] == user].head(1)['item_id'].values[0]
#     except IndexError:
#         most_i = random.choice(new_list)
#
#     try:
#         suggestions[i] = closestitems(itembyID(most_i))['id'].values
#     except ValueError:
#         j = j + 1
#         print("fuck me " + str(j) + " times")
#         continue




target = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)



def output(matrix):
    "Creates the file to be submitted given a kind of matrix"
    # output to file
    submission = open('submission2.csv', 'w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in matrix:
        line = str(row[0]) + ','
        i = 0
        for s in row[1]:
                if i == 0:
                    line = line + str(s)
                    i = 1
                else:
                    line = line + ' ' + str(s)
        submission.write(line + '\n')

    submission.close()
    return

def test(csv):
    test_set = pd.read_csv('test_set.csv', delimiter="\t")
    subm = pd.read_csv(csv, delimiter="\t")

