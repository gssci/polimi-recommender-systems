from project_utils import *
from sklearn import tree
from sklearn import utils

items = item_profiles[]
dio = tree.DecisionTreeRegressor()

for u in r_users:
    ids = items_id_raccomandabili(u)
    user_interactions = interagiti(u)
    X = items[items.id.isin(user_interactions)].drop('id',1).values
    y = np.zeros(user_interactions.size) + 1
    shuffled = utils.shuffle(X, y)
    X = shuffled[0]
    y = shuffled[1]
    dio = dio.fit(X,y)
    for i in ids:
        value = dio.predict(items[items['id'] == i].drop('id',1).values)
        print(str(i) + " - - " + str(value))


